"""
TPO Inference Strategy — compatible with SPO's treetune training framework.

Wraps ``build_tree_tpo`` (direct vLLM HTTP) so it can be used as a drop-in
replacement for SPO's guidance-based ``TreeInferenceStrategy``.

Output: a HuggingFace ``Dataset`` with column
    ``_treetune__reasoning_tree`` : str (JSON-serialised TPO Node tree)

This column is consumed by the episode generators (``TPOEpisodeGenerator``,
``TreeEpisodeGeneratorForMath``, etc.) exactly like SPO's trees.

Usage as a standalone object (no treetune required):

    strategy = TPOInferenceStrategy(
        server_url="http://localhost:8000/v1",
        model_name="my-model",
        branching_strategy=EntropyBranchingStrategy(min_branches=2, max_branches=8),
        termination_strategy=PDegradationTerminationStrategy(max_degradations=3),
        segmentation_strategy=FixedTokenSegmentation(max_tokens=512),
    )
    result_ds = strategy.generate(questions_dataset)

When treetune is importable, the class self-registers under the key ``"tpo"``
so it can be instantiated via jsonnet config:

    inference_strategy: {
        type: "tpo",
        server_url: "http://localhost:8000/v1",
        model_name: "...",
        ...
    }
"""

import json
import logging
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any, Callable, Dict, List, Optional

from datasets import Dataset

from mc_analysis._answer_utils import extract_answer as _default_extract_answer
from tpo.algorithm import build_tree_tpo
from tpo.strategies.branching import (
    BranchingStrategy,
    EntropyBranchingStrategy,
)
from tpo.strategies.termination import (
    TerminationStrategy,
    PDegradationTerminationStrategy,
)
from tpo.strategies.segmentation import (
    SegmentationStrategy,
    FixedTokenSegmentation,
)

logger = logging.getLogger(__name__)

TREE_COLNAME = "_treetune__reasoning_tree"

# ---------------------------------------------------------------------------
# Optional treetune registration
# ---------------------------------------------------------------------------

try:
    from treetune.inference_strategies.base_inference_strategy import (
        InferenceStrategy as _TreentuneBase,
    )

    _BASE = _TreentuneBase

    def _register(cls):
        _TreentuneBase.register("tpo", exist_ok=True)(cls)
        return cls

except ImportError:  # treetune not installed
    _BASE = object

    def _register(cls):
        return cls


@_register
class TPOInferenceStrategy(_BASE):
    """Inference strategy that builds TPO trees via direct vLLM HTTP requests.

    Parameters
    ----------
    server_url : str
        vLLM server base URL (e.g. ``"http://localhost:8000/v1"``).
    model_name : str
        Model identifier on the vLLM server.
    question_template : str
        Format string with a ``{query}`` placeholder that wraps each raw
        problem into a full prompt.  Mirrors SPO's ``question_template``.
    question_field : str
        Dataset column that contains the raw question / problem text.
        Default ``"query"``.
    answer_field : str
        Dataset column that contains the gold answer string (used to compute
        P(answer|trajectory) at every node).  Default ``"answer"``.
    branching_strategy : BranchingStrategy, optional
        Defaults to :class:`~tpo.strategies.branching.EntropyBranchingStrategy`.
    termination_strategy : TerminationStrategy, optional
        Defaults to :class:`~tpo.strategies.termination.PDegradationTerminationStrategy`.
    segmentation_strategy : SegmentationStrategy, optional
        Defaults to :class:`~tpo.strategies.segmentation.FixedTokenSegmentation`.
    extract_answer_fn : callable, optional
        ``text → answer_str | None``.  Defaults to
        :func:`mc_analysis._answer_utils.extract_answer`.
    get_logprobs : bool
        Whether to request per-token log-probs during generation (needed by
        some episode generators / the PPO trainer's prob-mask).
    top_k_entropy : int
        Number of top-K tokens fetched for entropy-based branching.
    max_workers : int
        Number of parallel tree-building workers (thread pool).  Each worker
        makes synchronous HTTP calls to the vLLM server.
    seed : int, optional
        RNG seed forwarded to vLLM.
    api_key : str, optional
        Bearer token for external OpenAI-compatible endpoints.
    """

    def __init__(
        self,
        server_url: Optional[str] = None,
        model_name: Optional[str] = None,
        question_template: str = "[MATH_TASK] Problem:\n{query}\n\nSolution:",
        question_field: str = "query",
        answer_field: str = "answer",
        branching_strategy: Optional[BranchingStrategy] = None,
        termination_strategy: Optional[TerminationStrategy] = None,
        segmentation_strategy: Optional[SegmentationStrategy] = None,
        extract_answer_fn: Optional[Callable[[str], Optional[str]]] = None,
        get_logprobs: bool = True,
        top_k_entropy: int = 20,
        max_workers: int = 4,
        seed: Optional[int] = None,
        api_key: Optional[str] = None,
        result_dir: Optional[Path] = None,
        cloud_logger=None,
        log_level=None,
    ):
        # Initialise base class if treetune is present
        if _BASE is not object:
            super().__init__(
                result_dir=result_dir,
                cloud_logger=cloud_logger,
                log_level=log_level,
            )

        self.server_url = server_url
        self.model_name = model_name
        self.question_template = question_template
        self.question_field = question_field
        self.answer_field = answer_field

        self.branching_strategy = branching_strategy or EntropyBranchingStrategy()
        self.termination_strategy = termination_strategy or PDegradationTerminationStrategy()
        self.segmentation_strategy = segmentation_strategy or FixedTokenSegmentation()
        self.extract_answer_fn = extract_answer_fn or _default_extract_answer

        self.get_logprobs = get_logprobs
        self.top_k_entropy = top_k_entropy
        self.max_workers = max_workers
        self.seed = seed
        self.api_key = api_key

    # ------------------------------------------------------------------
    # Internal helpers
    # ------------------------------------------------------------------

    def _prompt_for(self, example: Dict[str, Any]) -> str:
        query = example[self.question_field]
        return self.question_template.format(query=query)

    def _build_one(self, example: Dict[str, Any]) -> Dict[str, Any]:
        """Build a TPO tree for a single dataset example."""
        prompt = self._prompt_for(example)
        gold_answer = example[self.answer_field]

        root = build_tree_tpo(
            server_url=self.server_url,
            model_name=self.model_name,
            prompt=prompt,
            gold_answer=gold_answer,
            branching_strategy=self.branching_strategy,
            termination_strategy=self.termination_strategy,
            segmentation_strategy=self.segmentation_strategy,
            extract_answer_fn=self.extract_answer_fn,
            get_logprobs=self.get_logprobs,
            top_k_entropy=self.top_k_entropy,
            seed=self.seed,
            api_key=self.api_key,
            verbose=False,
        )

        result = dict(example)
        result[TREE_COLNAME] = json.dumps(root)
        return result

    # ------------------------------------------------------------------
    # Public interface (InferenceStrategy protocol)
    # ------------------------------------------------------------------

    def generate(self, dataset: Dataset) -> Dataset:
        """Build a TPO tree for every example and return the augmented dataset.

        Trees are built in parallel using a thread pool (``max_workers``
        concurrent vLLM sessions).  Results preserve dataset ordering.

        Parameters
        ----------
        dataset : Dataset
            Must contain at minimum the *question_field* and *answer_field*
            columns configured at construction time.

        Returns
        -------
        Dataset
            Input dataset with an additional ``_treetune__reasoning_tree``
            column (JSON string).
        """
        if self.server_url is None or self.model_name is None:
            raise RuntimeError(
                "[TPOInferenceStrategy] server_url and model_name must be set before "
                "calling generate(). In on-policy training these are injected "
                "automatically by TPOEpisodeGenerator._run_inference each iteration."
            )

        examples: List[Dict[str, Any]] = dataset.to_list()
        n = len(examples)
        ordered: List[Optional[Dict[str, Any]]] = [None] * n

        logger.info(f"[TPO] Building trees for {n} examples "
                    f"({self.max_workers} workers)...")

        with ThreadPoolExecutor(max_workers=self.max_workers) as pool:
            future_to_idx = {
                pool.submit(self._build_one, ex): i
                for i, ex in enumerate(examples)
            }
            done = 0
            for future in as_completed(future_to_idx):
                idx = future_to_idx[future]
                try:
                    ordered[idx] = future.result()
                except Exception as exc:
                    logger.error(f"[TPO] Example {idx} failed: {exc}", exc_info=True)
                    # Fallback: return example without tree column so downstream
                    # generators can skip it gracefully
                    ordered[idx] = dict(examples[idx])
                done += 1
                if done % max(1, n // 10) == 0:
                    logger.info(f"[TPO] {done}/{n} trees done")

        logger.info(f"[TPO] All {n} trees built.")
        return Dataset.from_list(ordered)
