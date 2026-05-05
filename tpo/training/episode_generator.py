"""
TPO Episode Generator — on-policy, SPO-compatible.

Inherits from SPO's MathEpisodeGenerator (which is itself an OnPolicyEpisodeGenerator)
so that the vLLM server is started/stopped internally with the CURRENT actor checkpoint
at the beginning of every training iteration — exactly as SPO does.

The key difference from SPO's MathEpisodeGeneratorWithMCAdvantages:
  - No MC rollouts for value estimation: values come directly from the TPO tree
    (P-degradation-terminated leaves get V=0; natural leaves get V=correctness).
  - `_run_inference` injects server_url / model_name at the top level of the
    inference strategy's lazy params, not inside a nested `guidance_llm` block.

V computation rules
-------------------
- Natural leaf (early_stopped=False): V = grade_answer(predicted, gold) ∈ {0, 1}
- Early-stopped leaf (early_stopped=True): V = 0.0 (P-degradation → hard zero)
- Internal node: V = mean(children V)
"""

import copy
import json
import logging
from dataclasses import asdict
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from mc_analysis._answer_utils import extract_answer as _default_extract_answer

logger = logging.getLogger(__name__)

TREE_COLNAME = "_treetune__reasoning_tree"

# ---------------------------------------------------------------------------
# Optional treetune registration
# ---------------------------------------------------------------------------

try:
    from accelerate.utils import release_memory
    from datasets import Dataset

    from treetune.common import Lazy
    from treetune.common.vllm_server import VLLMServer
    from treetune.episode_generators import EpisodeGenerator, MathEpisodeGenerator
    from treetune.episode_generators.base_episode_generator import Episode
    from treetune.tasks import Task
    from treetune.tokenization_utils import Tokenizer

    _HAS_TREETUNE = True

    @EpisodeGenerator.register("tpo", exist_ok=True)
    class TPOEpisodeGenerator(MathEpisodeGenerator):
        """TPO episode generator: on-policy, vLLM managed internally per iteration.

        Inherits from MathEpisodeGenerator (→ OnPolicyEpisodeGenerator) so
        the vLLM server is restarted with the latest actor checkpoint at the
        start of each PPO iteration — identical to SPO's behaviour.

        Overrides:
          _run_inference   — skip MC rollouts; inject server_url/model_name at top level
          _generate_episodes — convert TPO trees to episodes
          compute_score    — V=0 for P-degradation-terminated leaves
        """

        # ------------------------------------------------------------------
        # vLLM inference (replaces OnPolicyEpisodeGenerator._run_inference)
        # ------------------------------------------------------------------

        def _run_inference(
            self,
            dataset_shard: "Dataset",
            vllm_init_fn: Callable[[], Tuple["VLLMServer", Dict[str, Any]]],
            vllm_cleanup_fn: Callable[[], None],
            results_root_dir: Path,
            seed: int,
            iteration: int,
        ) -> "Dataset":
            """Start vLLM with current checkpoint, build TPO trees, stop vLLM.

            Unlike SPO's MathEpisodeGeneratorWithMCAdvantages, there are no MC
            rollouts: value estimates come from the tree structure itself.
            """
            tree_result_path = results_root_dir / "tpo_trees_ds"

            # 1. Start vLLM with the current actor checkpoint
            vllm_server, guidance_llm_kwargs = vllm_init_fn()
            # guidance_llm_kwargs = {"api_base": "http://host:port/v1",
            #                        "model":    "path/to/checkpoint"}

            # 2. Build TPO trees — inject server_url / model_name at top level
            #    (SPO's guidance_llm pattern does the same thing but nested)
            inference_strategy_lazy = copy.deepcopy(self.inference_strategy_lazy)
            inference_strategy_lazy._params["server_url"] = guidance_llm_kwargs["api_base"]
            inference_strategy_lazy._params["model_name"] = guidance_llm_kwargs["model"]

            inference_strategy = inference_strategy_lazy.construct(
                result_dir=results_root_dir,
                seed=seed,
                cloud_logger=None,
                log_level=(
                    logging.WARNING
                    if not self.distributed_state.is_local_main_process
                    else None
                ),
            )

            results = inference_strategy.generate(dataset_shard)
            results.save_to_disk(str(tree_result_path))

            # 3. Stop vLLM (mirrors OnPolicyEpisodeGenerator._run_inference)
            vllm_server.stop_server()
            del results
            del vllm_server
            del inference_strategy
            del inference_strategy_lazy
            release_memory()

            vllm_cleanup_fn()
            release_memory()

            return Dataset.load_from_disk(str(tree_result_path))

        # ------------------------------------------------------------------
        # Episode generation (from TPO trees → training episodes)
        # ------------------------------------------------------------------

        def _generate_episodes(
            self,
            inference_results: "Dataset",
            iteration: int,
        ) -> List[Union[Dict[str, Any], "Episode"]]:
            """Convert each TPO tree into one Episode per root-to-leaf path."""
            episodes: List[Dict[str, Any]] = []

            for row in inference_results:
                if TREE_COLNAME not in row:
                    logger.warning("[TPO] Row missing %s — skipping", TREE_COLNAME)
                    continue

                tree = json.loads(row[TREE_COLNAME])
                gold_answer = row.get("answer", "")

                # Bottom-up V computation (TPO rule: early-stopped → V=0)
                self.compute_score(self.task, tree, gold_answer)
                # Per-node advantage = V(node) - V(parent)
                self.compute_advantage(tree)

                for path in self._extract_tpo_paths(tree):
                    ep = self._tpo_path_to_episode(path)
                    if ep is not None:
                        episodes.append(asdict(ep))

            logger.info("[TPO] Generated %d episodes from %d trees",
                        len(episodes), len(inference_results))
            return episodes

        # ------------------------------------------------------------------
        # Score computation (overrides TreeEpisodeUtils.compute_score)
        # ------------------------------------------------------------------

        def is_answer_correct(
            self,
            task: "Task",
            node: Dict[str, Any],
            gold_answer: str,
        ) -> float:
            pred = self.extract_pred_answer(task, node)
            if pred is None:
                return 0.0
            return float(task.grade_answer(given_answer=pred, ground_truth=gold_answer))

        def compute_score(
            self,
            task: "Task",
            root: Dict[str, Any],
            gold_answer: str,
        ) -> Dict[str, Any]:
            """Bottom-up V with TPO early-stopping rule (V=0 for P-degradation leaves)."""
            def _dfs(node: Dict[str, Any]) -> float:
                if "answer" in node:
                    if node.get("early_stopped", False):
                        node["score"] = 0.0          # TPO spec
                    else:
                        node["score"] = self.is_answer_correct(task, node, gold_answer)
                    node["is_correct_answer"] = node["score"]
                    return node["score"]

                child_scores = [_dfs(c) for c in node.get("children", [])]
                node["score"] = float(np.mean(child_scores)) if child_scores else 0.0
                return node["score"]

            _dfs(root)
            return root

        # ------------------------------------------------------------------
        # Helpers
        # ------------------------------------------------------------------

        def _extract_tpo_paths(
            self,
            tree: Dict[str, Any],
        ) -> List[List[Dict[str, Any]]]:
            """Return all root-to-leaf paths (each path is a list of nodes)."""
            paths: List[List[Dict[str, Any]]] = []

            def _dfs(node: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
                current = chain + [node]
                if "answer" in node or not node.get("children"):
                    paths.append(current)
                    return
                for child in node.get("children", []):
                    _dfs(child, current)

            _dfs(tree, [])
            return paths

        def _tpo_path_to_episode(
            self,
            path: List[Dict[str, Any]],
        ) -> Optional["Episode"]:
            """Convert a root-to-leaf node chain to an Episode.

            Per-token advantages are assigned by distributing each node's
            advantage uniformly across all characters in that node's text,
            then mapping character → token via the tokenizer's offset_mapping.
            This is identical to SPO's advantage assignment.
            """
            if len(path) < 2:
                return None

            root = path[0]
            leaf = path[-1]

            query_text = root["full_text"]
            full_text = leaf["full_text"]
            response_text = full_text[len(query_text):]

            if not response_text:
                return None

            is_early_stopped = leaf.get("early_stopped", False)
            score = float(leaf.get("score", 0.0))

            try:
                query_ids, response_ids, offsets = self._tokenize_trajectory(
                    {"query_text": query_text, "response_text": response_text},
                    is_unfinished_response=is_early_stopped,
                    return_offsets=True,
                )
            except Exception as exc:
                logger.warning("[TPO] Tokenization failed: %s", exc)
                return None

            if not response_ids:
                return None

            # Build per-character advantage/value arrays over response_text
            char_adv = np.zeros(len(response_text), dtype=np.float32)
            char_val = np.zeros(len(response_text), dtype=np.float32)
            offset = 0
            for node in path[1:]:   # skip root (= query only, no text span)
                node_text = node.get("text", "")
                length = min(len(node_text), len(response_text) - offset)
                if length <= 0:
                    break
                char_adv[offset:offset + length] = node.get("advantage", 0.0)
                char_val[offset:offset + length] = node.get("score", 0.0)
                offset += length

            # offsets covers query+response; response starts at index len(query_ids)
            query_char_len = len(query_text)
            response_offsets = offsets[len(query_ids):]

            def _char_to_tok(arr: np.ndarray, char_offsets) -> np.ndarray:
                result = np.zeros(len(char_offsets), dtype=np.float32)
                for i, (start, _end) in enumerate(char_offsets):
                    rel = start - query_char_len
                    result[i] = arr[max(0, min(rel, len(arr) - 1))]
                return result

            adv_token = _char_to_tok(char_adv, response_offsets)
            val_token = _char_to_tok(char_val, response_offsets)

            return Episode(
                query_text=query_text,
                response_text=response_text,
                query_token_ids=query_ids,
                response_token_ids=response_ids,
                scores=score,
                advantages=adv_token.tolist(),
                values=val_token.tolist(),
            )

except ImportError:
    _HAS_TREETUNE = False
    TPOEpisodeGenerator = None  # type: ignore
    logger.warning(
        "[TPOEpisodeGenerator] treetune not available — "
        "TPOEpisodeGenerator will not be registered. "
        "Use the standalone helper functions below instead."
    )


# ---------------------------------------------------------------------------
# Standalone helpers (no treetune required)
# Used when running TPO outside the full SPO/treetune training stack.
# ---------------------------------------------------------------------------

def compute_score_tpo(
    root: Dict[str, Any],
    gold_answer: str,
    answer_checker: Callable[[Optional[str], str], bool],
    extract_answer_fn: Callable[[str], Optional[str]] = _default_extract_answer,
) -> Dict[str, Any]:
    """Bottom-up V computation (TPO rule) — standalone, no treetune needed.

    Leaf V:
        early_stopped=True  → 0.0
        natural EOS         → answer_checker(predicted, gold)
    Internal node V: mean(children V)
    """
    def _dfs(node: Dict[str, Any]) -> float:
        if "answer" in node:
            if node.get("early_stopped", False):
                node["score"] = 0.0
            else:
                pred = extract_answer_fn(node.get("answer", ""))
                node["score"] = float(
                    pred is not None and answer_checker(pred, gold_answer)
                )
            node["is_correct_answer"] = node["score"]
            return node["score"]

        child_scores = [_dfs(c) for c in node.get("children", [])]
        node["score"] = float(np.mean(child_scores)) if child_scores else 0.0
        return node["score"]

    _dfs(root)
    return root


def compute_advantage_tpo(root: Dict[str, Any]) -> Dict[str, Any]:
    """Per-node advantage = V(node) - V(parent). Stored in node['advantage']."""
    def _dfs(node: Dict[str, Any]) -> None:
        if "answer" in node:
            return
        for child in node.get("children", []):
            child["advantage"] = child["score"] - node["score"]
            _dfs(child)

    _dfs(root)
    return root


def extract_paths_tpo(
    root: Dict[str, Any],
) -> List[List[Dict[str, Any]]]:
    """Return all root-to-leaf paths from a TPO tree."""
    paths: List[List[Dict[str, Any]]] = []

    def _dfs(node: Dict[str, Any], chain: List[Dict[str, Any]]) -> None:
        current = chain + [node]
        if "answer" in node or not node.get("children"):
            paths.append(current)
            return
        for child in node.get("children", []):
            _dfs(child, current)

    _dfs(root, [])
    return paths


def tree_to_episodes_standalone(
    tree_json: str,
    gold_answer: str,
    tokenizer,
    answer_checker: Callable[[Optional[str], str], bool],
    extract_answer_fn: Callable[[str], Optional[str]] = _default_extract_answer,
    append_eos: bool = True,
) -> List[Dict[str, Any]]:
    """Convert a JSON-serialised TPO tree into training episode dicts.

    Standalone alternative to ``TPOEpisodeGenerator`` for use outside the
    treetune framework (e.g. with verl / TRL).

    Each episode dict contains:
        query_token_ids    : List[int]
        response_token_ids : List[int]
        advantages         : List[float]  (per-token)
        values             : List[float]  (per-token V(parent))
        score              : float
        early_stopped      : bool
        query_text         : str
        response_text      : str
    """
    root = json.loads(tree_json)
    compute_score_tpo(root, gold_answer, answer_checker, extract_answer_fn)
    compute_advantage_tpo(root)

    paths = extract_paths_tpo(root)
    episodes = []

    for path in paths:
        if len(path) < 2:
            continue

        query_text = path[0]["full_text"]
        leaf = path[-1]
        full_text = leaf["full_text"]
        response_text = full_text[len(query_text):]
        is_early_stopped = leaf.get("early_stopped", False)
        score = float(leaf.get("score", 0.0))

        if not response_text:
            continue

        # Tokenise with offset mapping
        full_enc = tokenizer(
            f"{query_text}{response_text}",
            return_offsets_mapping=True,
            add_special_tokens=False,
        )
        query_enc = tokenizer(query_text, add_special_tokens=False)
        n_query = len(query_enc["input_ids"])

        full_ids = full_enc["input_ids"]
        response_ids = full_ids[n_query:]
        response_offsets = full_enc["offset_mapping"][n_query:]

        if not response_ids:
            continue

        # Per-character advantage / value
        char_adv = np.zeros(len(response_text), dtype=np.float32)
        char_val = np.zeros(len(response_text), dtype=np.float32)
        off = 0
        for node in path[1:]:
            t = node.get("text", "")
            ln = min(len(t), len(response_text) - off)
            if ln <= 0:
                break
            char_adv[off:off + ln] = node.get("advantage", 0.0)
            char_val[off:off + ln] = node.get("score", 0.0)
            off += ln

        query_char_len = len(query_text)

        def _map(arr):
            return np.array(
                [arr[max(0, min(s - query_char_len, len(arr) - 1))]
                 for s, _ in response_offsets],
                dtype=np.float32,
            )

        adv_tok = _map(char_adv)
        val_tok = _map(char_val)

        if append_eos and not is_early_stopped:
            eos = getattr(tokenizer, "eos_token_id", None)
            if eos is not None:
                response_ids = list(response_ids) + [eos]
                adv_tok = np.append(adv_tok, 0.0)
                val_tok = np.append(val_tok, 0.0)

        episodes.append({
            "query_text": query_text,
            "response_text": response_text,
            "query_token_ids": list(full_ids[:n_query]),
            "response_token_ids": list(response_ids),
            "advantages": adv_tok.tolist(),
            "values": val_tok.tolist(),
            "score": score,
            "early_stopped": bool(is_early_stopped),
        })

    return episodes
