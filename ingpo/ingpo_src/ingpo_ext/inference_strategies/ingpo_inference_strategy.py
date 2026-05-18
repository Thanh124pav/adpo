"""InGPO inference strategy: SPO-tree with online Share / Prune triggers.

Subclasses `HybridInferenceStrategy` from SPO and overrides `_construct_tree`
so that for every freshly-expanded child segment we:

  1. Compute K fast logprobs `log pi(y_i | traj(child))` against the
     per-problem answer set Y.
  2. Consult the `TriggerEngine` for SHARE / PRUNE / EXPAND.
  3. Annotate the child node with `ingpo_action`, `ingpo_share_target`,
     `ingpo_avg_lp_K`, `ingpo_avg_lp_m`, `ingpo_tv_m`.
  4. Skip recursion into SHARE / PRUNE children — but keep them in the tree
     because the downstream episode generator still consumes them as edges.

The answer set Y is built once per problem at `depth==1` (i.e. before the
first batch of root children is expanded), in parallel with that expansion.

All other behaviour — branch_factor strategy, segment cap M, reward
function, full_text accounting — is inherited unchanged from SPO so the
codepath remains identical to a faithful SPO-tree run when triggers are
disabled.
"""

from __future__ import annotations

import asyncio
import logging
import os
import time
import uuid
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import openai

from treetune.common import Lazy
from treetune.inference_strategies.base_inference_strategy import InferenceStrategy
from treetune.inference_strategies.hybrid_inference_strategy import (
    HybridInferenceStrategy,
)
from treetune.inference_strategies.tree_inference import Node
from treetune.logging_utils import get_logger

from ingpo_ext.core.answer_set import (
    DEFAULT_Y_PROMPT_TEMPLATE,
    AnswerSet,
    AnswerSetGenerator,
)
from ingpo_ext.core.logging_helpers import aggregate_tree_stats
from ingpo_ext.core.thresholds import ThresholdConfig
from ingpo_ext.core.triggers import Action, TriggerEngine
from ingpo_ext.core.local_value_share import (
    LocalShareDecision,
    confidence_radius,
    pair_budget,
    sampled_tv_from_logps,
    select_candidate_pairs,
    stable_softmax,
)
from ingpo_ext.core.vllm_scorer import VLLMLogprobClient, make_lp_scorer

logger = get_logger(__name__)


@InferenceStrategy.register("ingpo", exist_ok=True)
class InGPOInferenceStrategy(HybridInferenceStrategy):
    def __init__(
        self,
        # InGPO-specific knobs ------------------------------------------------
        ingpo_K: int = 10,
        ingpo_m: int = 100,
        ingpo_epsilon: float = 0.02,
        ingpo_r_max: float = 1.0,
        ingpo_alpha: float = 0.05,
        ingpo_use_dkw: bool = True,
        ingpo_eta_override: Optional[float] = None,
        ingpo_enable_share: bool = True,
        ingpo_enable_prune: bool = True,
        ingpo_share_target: str = "nearest",
        ingpo_local_value_share: bool = True,
        ingpo_share_pair_budget_fraction: float = 0.25,
        ingpo_share_use_confidence: bool = False,
        ingpo_y_prompt_template: str = DEFAULT_Y_PROMPT_TEMPLATE,
        ingpo_y_temperature: float = 0.7,
        ingpo_y_max_tokens: int = 512,
        ingpo_y_field: str = "answer",  # field on data_instance with gold
        ingpo_score_concurrency: int = 64,
        # Inherited ----------------------------------------------------------
        **kwargs: Any,
    ):
        super().__init__(**kwargs)
        self.cfg_thresholds = ThresholdConfig(
            epsilon=ingpo_epsilon,
            r_max=ingpo_r_max,
            alpha=ingpo_alpha,
            K=ingpo_K,
            use_dkw=ingpo_use_dkw,
            eta_override=ingpo_eta_override,
        )
        self.ingpo_m = int(ingpo_m)
        self.ingpo_enable_share = bool(ingpo_enable_share)
        self.ingpo_enable_prune = bool(ingpo_enable_prune)
        self.ingpo_share_target = ingpo_share_target
        self.ingpo_local_value_share = bool(ingpo_local_value_share)
        self.ingpo_share_pair_budget_fraction = float(ingpo_share_pair_budget_fraction)
        self.ingpo_share_use_confidence = bool(ingpo_share_use_confidence)
        self.ingpo_y_prompt_template = ingpo_y_prompt_template
        self.ingpo_y_temperature = float(ingpo_y_temperature)
        self.ingpo_y_max_tokens = int(ingpo_y_max_tokens)
        self.ingpo_y_field = ingpo_y_field
        self.ingpo_score_concurrency = int(ingpo_score_concurrency)
        self._lp_client: Optional[VLLMLogprobClient] = None

    # ------------------------------------------------------------------
    # vLLM scoring client
    # ------------------------------------------------------------------

    def _ensure_lp_client(self) -> VLLMLogprobClient:
        if self._lp_client is not None:
            return self._lp_client
        guidance_kwargs = self.guidance_llm_lazy._params  # type: ignore[attr-defined]
        api_base = guidance_kwargs.get("api_base")
        if api_base in (None, "none"):
            api_base = os.environ.get("APP_OPENAI_VLLM_API_BASE")
        if api_base is None:
            raise RuntimeError("InGPO needs api_base to call vLLM /completions")
        model = guidance_kwargs.get("model")
        api_key = guidance_kwargs.get("api_key", "EMPTY")
        self._lp_client = VLLMLogprobClient(
            api_base=api_base,
            model=model,
            api_key=api_key,
            max_concurrency=self.ingpo_score_concurrency,
        )
        return self._lp_client

    def _tokenize(self, text: str) -> List[int]:
        if self.tokenizer is None:
            raise RuntimeError("InGPO requires a tokenizer to count suffix tokens")
        return self.tokenizer(text).input_ids

    # ------------------------------------------------------------------
    # Build per-problem Y
    # ------------------------------------------------------------------

    async def _build_answer_set(
        self,
        problem_id: str,
        problem_text: str,
        gold: str,
    ) -> AnswerSet:
        client = openai.AsyncOpenAI(
            api_key=self._lp_client.api_key,
            base_url=self._lp_client.api_base,
        ) if self._lp_client is not None else None

        async def sample_fn(prompt: str, n: int, temperature: float, max_tokens: int):
            if client is None:
                return []
            resp = await client.completions.create(
                model=self._lp_client.model,
                prompt=prompt,
                n=n,
                temperature=temperature,
                max_tokens=max_tokens,
            )
            return [c.text for c in resp.choices]

        gen = AnswerSetGenerator(
            sample_fn=sample_fn,
            m=self.ingpo_m,
            temperature=self.ingpo_y_temperature,
            max_tokens=self.ingpo_y_max_tokens,
            prompt_template=self.ingpo_y_prompt_template,
        )
        try:
            return await gen.build(problem_id=problem_id, problem=problem_text, gold=gold)
        except Exception as exc:
            logger.warning(f"Y generation failed for problem {problem_id}: {exc}")
            return AnswerSet(problem_id=problem_id, gold=gold, y=[])

    # ------------------------------------------------------------------
    # Override tree construction
    # ------------------------------------------------------------------

    async def _construct_tree(
        self,
        initial_prompt: str,
        max_depth: int = 2,
        data_instance: Optional[Dict[str, Any]] = None,
    ):
        t0_tree = time.time()
        client = self._ensure_lp_client()
        scorer = make_lp_scorer(client, self._tokenize)

        problem_text = data_instance.get("problem") if data_instance else None
        gold = data_instance.get(self.ingpo_y_field) if data_instance else None
        problem_id = str(data_instance.get("_treetune__idx", uuid.uuid4())) if data_instance else str(uuid.uuid4())

        needs_answer_set = self.ingpo_enable_share and not self.ingpo_local_value_share

        # The sibling-local ValueShare / Prune path does not need Y.  We only
        # build Y for the legacy answer-set ValueShare trigger.
        answer_set: AnswerSet = AnswerSet(problem_id=problem_id, gold=gold or "", y=[])
        y_task = (
            asyncio.create_task(
                self._build_answer_set(problem_id, problem_text or "", gold or "")
            )
            if needs_answer_set
            else None
        )

        engine: Optional[TriggerEngine] = None
        local_shared_count = 0
        local_avg_tv_share = 0.0
        local_pruned_count = 0
        local_avg_gap_prune = 0.0

        tree: Node = {
            "text": initial_prompt,
            "depth": 0,
            "full_text": initial_prompt,
            "stop_text": "aaa",
            "_request_object": data_instance,
            "leaf": False,
            "ingpo_action": Action.EXPAND.value,
            "ingpo_segment_id": "root",
        }

        async def _ensure_engine() -> Optional[TriggerEngine]:
            nonlocal engine, answer_set
            if not needs_answer_set or y_task is None:
                return None
            if engine is not None:
                return engine
            answer_set = await y_task
            if answer_set.m == 0:
                logger.warning(
                    "InGPO: empty answer set; falling back to vanilla SPO-tree expansion"
                )
                return None
            engine = TriggerEngine(
                answer_set=answer_set,
                scorer=scorer,
                thresholds=self.cfg_thresholds,
                enable_share=self.ingpo_enable_share and not self.ingpo_local_value_share,
                enable_prune=False,
                share_target=self.ingpo_share_target,
                root_segment_id="root",
            )
            await engine.register_root(initial_prompt)
            return engine

        def _share_eta() -> float:
            if self.cfg_thresholds.eta_override is not None:
                return float(self.cfg_thresholds.eta_override)
            return max(
                self.cfg_thresholds.epsilon / max(self.cfg_thresholds.r_max, 1e-8),
                1e-6,
            )

        def _cheap_share_score(child: Node) -> float:
            val = child.get("ingpo_avg_lp_K")
            if val is not None:
                return float(val)
            return float(len(self._tokenize(child.get("text", ""))))

        def _continuation_texts(node: Node) -> List[str]:
            texts: List[str] = []
            seen = set()
            for child in node.get("children", []) or []:
                text = child.get("text", "")
                if not text or text in seen:
                    continue
                seen.add(text)
                texts.append(text)
                if len(texts) >= self.ingpo_m:
                    break
            return texts

        async def _score_local_tv(
            src: Node,
            tgt: Node,
            continuations: Sequence[str],
        ) -> float:
            src_prefix = src["full_text"]
            tgt_prefix = tgt["full_text"]
            src_scores, tgt_scores = await asyncio.gather(
                asyncio.gather(*(scorer.score_one(src_prefix, c) for c in continuations)),
                asyncio.gather(*(scorer.score_one(tgt_prefix, c) for c in continuations)),
            )
            return sampled_tv_from_logps(src_scores, tgt_scores)

        async def _score_sibling_probs(parent: Node, siblings: Sequence[Node]) -> Dict[str, float]:
            if not siblings:
                return {}
            scores = await asyncio.gather(
                *(scorer.score_one(parent["full_text"], child.get("text", "")) for child in siblings)
            )
            probs = stable_softmax(scores)
            return {
                child["ingpo_segment_id"]: float(probs[idx])
                for idx, child in enumerate(siblings)
            }

        async def _try_local_value_share_and_prune(parent: Node, siblings: Sequence[Node]) -> None:
            nonlocal local_shared_count, local_avg_tv_share
            nonlocal local_pruned_count, local_avg_gap_prune
            candidates = [
                child for child in siblings
                if child.get("ingpo_action") == Action.EXPAND.value
                and not child.get("leaf", False)
                and child.get("children")
            ]
            if len(candidates) < 2:
                return

            budget = pair_budget(
                len(candidates),
                fraction=self.ingpo_share_pair_budget_fraction,
            )
            pairs = select_candidate_pairs(
                [c["ingpo_segment_id"] for c in candidates],
                [_cheap_share_score(c) for c in candidates],
                budget=budget,
            )
            eta = _share_eta()
            pair_tvs: Dict[frozenset, float] = {}

            for i, j in pairs:
                src = candidates[j]
                tgt = candidates[i]

                continuations = []
                seen = set()
                for text in _continuation_texts(src) + _continuation_texts(tgt):
                    if text in seen:
                        continue
                    seen.add(text)
                    continuations.append(text)
                    if len(continuations) >= self.ingpo_m:
                        break
                if not continuations:
                    continue

                try:
                    tv = await _score_local_tv(src, tgt, continuations)
                except Exception as exc:
                    logger.warning(f"InGPO local ValueShare failed: {exc}")
                    continue

                pair_tvs[frozenset((src["ingpo_segment_id"], tgt["ingpo_segment_id"]))] = tv
                radius = confidence_radius(len(continuations), self.cfg_thresholds.alpha)
                lhs = tv + radius if self.ingpo_share_use_confidence else tv
                if src.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if tgt.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if lhs <= eta:
                    decision = LocalShareDecision(
                        source_id=src["ingpo_segment_id"],
                        target_id=tgt["ingpo_segment_id"],
                        tv=tv,
                        n_continuations=len(continuations),
                        confidence_radius=radius,
                        eta_used=eta,
                    )
                    src["ingpo_action"] = Action.SHARE.value
                    src["ingpo_share_target"] = decision.target_id
                    src["ingpo_tv_m"] = decision.tv
                    src["ingpo_local_support_size"] = decision.n_continuations
                    src["ingpo_confidence_radius"] = decision.confidence_radius
                    src["ingpo_eta"] = decision.eta_used
                    src["ingpo_tau"] = decision.confidence_radius
                    src["ingpo_local_value_share"] = True
                    src["leaf"] = True
                    src["reward"] = float("nan")
                    src.pop("children", None)
                    if engine is not None:
                        if engine.stats.expanded > 0:
                            engine.stats.expanded -= 1
                        engine.stats.update_share(tv)
                    else:
                        n = local_shared_count + 1
                        local_avg_tv_share = (
                            local_avg_tv_share * local_shared_count + tv
                        ) / n
                        local_shared_count = n

            if not self.ingpo_enable_prune:
                return

            try:
                sibling_probs = await _score_sibling_probs(parent, candidates)
            except Exception as exc:
                logger.warning(f"InGPO local Prune probability scoring failed: {exc}")
                return

            for child in candidates:
                if child.get("ingpo_action") != Action.EXPAND.value:
                    continue
                if child.get("leaf", False):
                    continue
                # Never prune depth-1 nodes.  The root's first branching layer
                # is too early for a stable value comparison.
                if int(child.get("ingpo_depth", 0) or 0) <= 1:
                    continue

                weighted_bound = 0.0
                child_id = child["ingpo_segment_id"]
                for other in candidates:
                    other_id = other["ingpo_segment_id"]
                    if other_id == child_id:
                        continue
                    tv = pair_tvs.get(frozenset((child_id, other_id)), 1.0)
                    weighted_bound += (
                        sibling_probs.get(other_id, 0.0)
                        * self.cfg_thresholds.r_max
                        * tv
                    )

                if weighted_bound < self.cfg_thresholds.epsilon:
                    child["ingpo_action"] = Action.PRUNE.value
                    child["ingpo_share_target"] = None
                    child["ingpo_value_parent_gap_bound"] = weighted_bound
                    child["ingpo_prune_value_eps"] = self.cfg_thresholds.epsilon
                    child["ingpo_prune_policy_prob"] = sibling_probs.get(child_id, 0.0)
                    child["leaf"] = True
                    child["reward"] = float("nan")
                    child.pop("children", None)

                    if engine is not None:
                        if engine.stats.expanded > 0:
                            engine.stats.expanded -= 1
                        engine.stats.update_prune(weighted_bound)
                    else:
                        n = local_pruned_count + 1
                        local_avg_gap_prune = (
                            local_avg_gap_prune * local_pruned_count + weighted_bound
                        ) / n
                        local_pruned_count = n

        def _set_reward_summary(node: Node) -> None:
            child_rewards = []
            for ch in node.get("children", []) or []:
                r = ch.get("reward")
                if r is None or (isinstance(r, float) and np.isnan(r)):
                    continue
                child_rewards.append(r)
            if child_rewards:
                node["reward"] = float(np.mean(child_rewards))
                node["reward_std"] = float(np.std(child_rewards))
            else:
                node["reward"] = 0.0
                node["reward_std"] = 0.0

        async def dfs(node: Node, prefix: str, depth: int) -> None:
            if depth == max_depth:
                node["reward"], _ = self.reward_function(
                    query=prefix,
                    response=node["text"],
                    dataset_instance=data_instance,
                )
                node["leaf"] = True
                return

            max_tokens = None if depth == max_depth - 1 else self.M

            children = node.get("children")
            if children is None:
                children = await self.node_expander.expand(
                    current_node=node,
                    prefix=prefix,
                    depth=depth,
                    max_tokens=max_tokens,
                )
            node["children"] = children

            local_engine = await _ensure_engine()

            probe_tasks = []
            for ch_idx, child in enumerate(children):
                child_seg_id = child.get(
                    "ingpo_segment_id",
                    f"{node.get('ingpo_segment_id', 'root')}/{depth}/{ch_idx}",
                )
                child["ingpo_segment_id"] = child_seg_id
                child["ingpo_action"] = Action.EXPAND.value
                child["ingpo_depth"] = depth + 1
                child["ingpo_parent_segment_id"] = node.get("ingpo_segment_id", "root")
                if child["finish_reason"] != "length":
                    child["reward"], _ = self.reward_function(
                        query=prefix,
                        response=child["full_text"],
                        dataset_instance=data_instance,
                    )
                    child["leaf"] = True
                    if local_engine is not None:
                        try:
                            decision = await local_engine.decide(
                                segment_id=child_seg_id,
                                parent_id=node.get("ingpo_segment_id", "root"),
                                prefix=child["full_text"],
                                is_leaf=True,
                            )
                            self._annotate_node(child, decision)
                        except Exception as exc:
                            logger.warning(f"InGPO decide() failed (leaf): {exc}")
                    continue

                child["leaf"] = False
                if local_engine is None:
                    if depth + 1 < max_depth:
                        probe_tasks.append((child, asyncio.create_task(
                            self.node_expander.expand(
                                current_node=child,
                                prefix=child["full_text"],
                                depth=depth + 1,
                                max_tokens=None if depth + 1 == max_depth - 1 else self.M,
                            )
                        )))
                    continue

                try:
                    decision = await local_engine.decide(
                        segment_id=child_seg_id,
                        parent_id=node.get("ingpo_segment_id", "root"),
                        prefix=child["full_text"],
                        is_leaf=False,
                    )
                except Exception as exc:
                    logger.warning(f"InGPO decide() failed: {exc}")
                    if depth + 1 < max_depth:
                        probe_tasks.append((child, asyncio.create_task(
                            self.node_expander.expand(
                                current_node=child,
                                prefix=child["full_text"],
                                depth=depth + 1,
                                max_tokens=None if depth + 1 == max_depth - 1 else self.M,
                            )
                        )))
                    continue

                self._annotate_node(child, decision)

                if decision.action is Action.EXPAND:
                    if depth + 1 < max_depth:
                        probe_tasks.append((child, asyncio.create_task(
                            self.node_expander.expand(
                                current_node=child,
                                prefix=child["full_text"],
                                depth=depth + 1,
                                max_tokens=None if depth + 1 == max_depth - 1 else self.M,
                            )
                        )))
                elif decision.action is Action.SHARE:
                    # Inherit value from share_target (set later in episode
                    # generator) but mark as a leaf so the tree code stops.
                    child["leaf"] = True
                    child["reward"] = float("nan")  # sentinel; replaced later
                else:  # Action.PRUNE
                    child["leaf"] = True
                    child["reward"] = float("nan")  # sentinel; replaced later

            for child, task in probe_tasks:
                try:
                    probe_children = await task
                except Exception as exc:
                    logger.warning(f"InGPO probe expansion failed: {exc}")
                    continue
                for probe_idx, probe in enumerate(probe_children):
                    probe["ingpo_segment_id"] = (
                        f"{child['ingpo_segment_id']}/{depth + 1}/{probe_idx}"
                    )
                    probe["ingpo_action"] = Action.EXPAND.value
                    probe["ingpo_depth"] = depth + 2
                    probe["ingpo_parent_segment_id"] = child["ingpo_segment_id"]
                child["children"] = probe_children

            if self.ingpo_local_value_share and (self.ingpo_enable_share or self.ingpo_enable_prune):
                await _try_local_value_share_and_prune(node, children)

            expansion_tasks = []
            for child in children:
                if child.get("ingpo_action") == Action.EXPAND.value and not child.get("leaf", False):
                    expansion_tasks.append(
                        asyncio.create_task(dfs(child, child["full_text"], depth + 1))
                    )

            if expansion_tasks:
                await asyncio.gather(*expansion_tasks)

            # Compute reward / reward_std *after* descendants finish.  For
            # SHARE / PRUNE children, we postpone the value until the episode
            # generator can resolve them; for now use the parent's prior
            # average to keep tree-level statistics finite.
            _set_reward_summary(node)

        await dfs(tree, initial_prompt, 0)

        # Aggregate stats from the final tree once.  Same tree walk as
        # ``per_depth_action_counts`` → aggregate rates can't disagree with
        # per-depth rates, and the local_value_share path stops dropping
        # expanded nodes from the denominator.
        stats = aggregate_tree_stats(tree)
        if stats:
            stats["ingpo/avg_tv_when_share"] = (
                engine.stats.avg_tv_share if engine is not None else local_avg_tv_share
            )
            stats["ingpo/avg_gap_when_prune"] = (
                engine.stats.avg_gap_prune if engine is not None else local_avg_gap_prune
            )
        tree["ingpo_stats"] = stats
        tree["ingpo_answer_set_size"] = answer_set.m
        tree["ingpo_tree_construction_seconds"] = time.time() - t0_tree
        tree["ingpo_problem_id"] = problem_id
        return tree

    @staticmethod
    def _annotate_node(child: Node, decision) -> None:
        child["ingpo_action"] = decision.action.value
        child["ingpo_share_target"] = decision.share_target
        child["ingpo_avg_lp_K"] = decision.avg_lp_K
        if decision.avg_lp_m is not None:
            child["ingpo_avg_lp_m"] = decision.avg_lp_m
        if decision.tv_m is not None:
            child["ingpo_tv_m"] = decision.tv_m
        if decision.avg_lp_diff_to_pa_m is not None:
            child["ingpo_gap_m"] = decision.avg_lp_diff_to_pa_m
        child["ingpo_eta"] = decision.eta_used
        child["ingpo_tau"] = decision.tau_used
