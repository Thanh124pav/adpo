"""
Termination strategies for TPO.

A TerminationStrategy decides whether a node should be treated as a leaf
(not expanded further), based on the full path from the root to that node.
"""

import math
import random
from abc import ABC, abstractmethod
from typing import Any, Dict, List, Optional

import numpy as np

Node = Dict[str, Any]


class TerminationStrategy(ABC):
    """Abstract base for node-termination strategies.

    :meth:`should_terminate` is called for every freshly sampled child node
    *before* deciding whether to expand it.  It receives the complete path
    from root to the candidate node (inclusive) so that strategies can
    inspect ancestry.

    If the method returns ``True``, the node is treated as a leaf: an answer
    is extracted from its text and the DFS stops.  Natural EOS / stop-string
    termination is handled separately by the algorithm and is *not* routed
    through this method.
    """

    @abstractmethod
    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        """Return ``True`` if *node* should not be expanded further.

        Parameters
        ----------
        node : Node
            The candidate node (deepest element of *path*).
        path : List[Node]
            Full path ``[root, …, parent, node]`` from root to *node*,
            inclusive.  Every node in the path has already been sampled
            and annotated with at least ``"depth"`` and, when relevant,
            ``"P"`` (log P(answer|trajectory)).
        """


class DepthTerminationStrategy(TerminationStrategy):
    """Terminate when a node's depth reaches *max_depth* — same as SPO.

    Parameters
    ----------
    max_depth : int
        Nodes at this depth (or deeper) are not expanded further.
    """

    def __init__(self, max_depth: int = 4):
        self.max_depth = max_depth

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        return node.get("depth", len(path) - 1) >= self.max_depth


class PDegradationTerminationStrategy(TerminationStrategy):
    """Terminate when P(answer|trajectory) degrades too many times.

    **Core idea** (TPO spec):

    For the path ``[n_0 = root, n_1, …, n_k = node]``, count the number of
    nodes ``n_i`` (i ≥ 1) where

        P(answer | trajectory to n_i) < P(answer | trajectory to parent n_{i-1})

    i.e. where the model's confidence in the gold answer *decreased* after
    incorporating the reasoning step that produced ``n_i``.

    If this **degradation count exceeds K** (*max_degradations*), the node is
    treated as a terminal leaf — further reasoning along this branch is
    unlikely to reach the correct answer.

    A hard *max_depth* cap is also enforced as a safety backstop.

    .. note::
        ``node["P"]`` (log P of the gold answer given the trajectory up to
        that node) must be populated for every node in *path* before this
        strategy is called.  :func:`~tpo.algorithm.build_tree_tpo` handles
        this automatically by computing P inline for each sampled node.

    Parameters
    ----------
    max_degradations : int
        Threshold K.  A node is terminal when the degradation count
        **strictly exceeds** K (i.e. count > K).
    max_depth : int
        Hard depth cap applied independently of P degradation.
    """

    def __init__(self, max_degradations: int = 3, max_depth: int = 10):
        if max_degradations < 0:
            raise ValueError("max_degradations must be ≥ 0")
        self.K = max_degradations
        self.max_depth = max_depth

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        if node.get("depth", len(path) - 1) >= self.max_depth:
            return True

        # Count P degradations along path (skip root at index 0)
        degradations = 0
        for i in range(1, len(path)):
            p_curr = path[i].get("P")
            p_par = path[i - 1].get("P")
            if p_curr is not None and p_par is not None and p_curr < p_par:
                degradations += 1

        return degradations > self.K

    def degradation_count(self, path: List[Node]) -> int:
        """Utility: return the current degradation count for a given path."""
        count = 0
        for i in range(1, len(path)):
            p_curr = path[i].get("P")
            p_par = path[i - 1].get("P")
            if p_curr is not None and p_par is not None and p_curr < p_par:
                count += 1
        return count


class CombinedTerminationStrategy(TerminationStrategy):
    """Terminate when **any** of the supplied strategies says to terminate.

    Useful for layering multiple stopping conditions, e.g. combining a depth
    cap with P-degradation:

    .. code-block:: python

        strategy = CombinedTerminationStrategy([
            DepthTerminationStrategy(max_depth=6),
            PDegradationTerminationStrategy(max_degradations=2),
        ])

    Parameters
    ----------
    strategies : list of TerminationStrategy
        The strategies to consult.  The first one that returns ``True``
        short-circuits evaluation.
    """

    def __init__(self, strategies: List[TerminationStrategy]):
        if not strategies:
            raise ValueError("strategies must be non-empty")
        self.strategies = strategies

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        return any(s.should_terminate(node, path) for s in self.strategies)


class MLClassifierTermination(TerminationStrategy):
    """Termination driven by a learned classifier over P(answer|trajectory).

    **Core idea**: A lightweight sklearn classifier (logistic regression, SVM,
    or decision tree) maps the scalar ``exp(node["P"])`` ∈ [0, 1] to a
    termination probability.  The classifier is pre-trained on a small labelled
    dataset (collected before the main RL loop) and then held fixed throughout
    training.

    **Stochastic termination**: During training, nodes are terminated with
    probability ``p_terminate`` returned by the classifier rather than using a
    hard threshold.  This preserves exploration while still biasing towards
    pruning low-P paths.

    **Fallback**: If the classifier has not been trained yet (``fit`` was never
    called), the strategy falls back to a hard depth cap
    (``max_depth_fallback``).

    .. note::
        The classifier is intentionally simple (1-D input) so that it cannot
        overfit and its inductive bias is transparent.  The key assumption is
        that P(answer|traj) is a sufficient statistic for "should we continue?"
        — a consequence of the Optional Stopping Theorem / Doob's martingale
        argument (if the trajectory is a supermartingale, the optimal stopping
        rule depends only on the current value).

    Parameters
    ----------
    classifier_type : {'logistic', 'svm', 'decision_tree'}
        Which sklearn estimator to use.
    max_depth_fallback : int
        Hard depth cap used when the classifier is untrained.
    use_stochastic : bool
        If ``True``, terminate with probability ``p_terminate`` (training);
        if ``False``, use a hard 0.5 threshold (deterministic evaluation).
    class_weight : str or dict
        Passed to the sklearn estimator.  ``'balanced'`` compensates for
        imbalanced terminate/continue labels in the pre-training set.
    random_state : int, optional
        RNG seed for reproducibility of stochastic decisions and classifiers.
    """

    _VALID_CLASSIFIERS = {"logistic", "svm", "decision_tree"}

    def __init__(
        self,
        classifier_type: str = "logistic",
        max_depth_fallback: int = 10,
        use_stochastic: bool = True,
        class_weight: str = "balanced",
        random_state: Optional[int] = None,
    ):
        if classifier_type not in self._VALID_CLASSIFIERS:
            raise ValueError(
                f"classifier_type must be one of {self._VALID_CLASSIFIERS}, "
                f"got '{classifier_type}'"
            )
        self.classifier_type = classifier_type
        self.max_depth_fallback = max_depth_fallback
        self.use_stochastic = use_stochastic
        self.class_weight = class_weight
        self.random_state = random_state
        self.classifier = None
        self._rng = random.Random(random_state)

    # ------------------------------------------------------------------
    # Training interface
    # ------------------------------------------------------------------

    def fit(self, P_values: List[float], labels: List[int]) -> None:
        """Pre-train the classifier.

        Parameters
        ----------
        P_values : list of float
            exp(log_P) values ∈ [0, 1] collected from a small dataset.
            Each entry corresponds to the P value at a node in a trajectory.
        labels : list of int
            Binary labels: ``1`` = should terminate (low-P / dead-end path),
            ``0`` = should continue expanding.
        """
        from sklearn.linear_model import LogisticRegression
        from sklearn.svm import SVC
        from sklearn.tree import DecisionTreeClassifier

        X = np.array(P_values, dtype=np.float64).reshape(-1, 1)
        y = np.array(labels, dtype=int)

        if len(np.unique(y)) < 2:
            raise ValueError(
                "Training labels must contain both classes (0 and 1). "
                f"Got only label(s): {np.unique(y).tolist()}"
            )

        if self.classifier_type == "logistic":
            clf = LogisticRegression(
                class_weight=self.class_weight,
                random_state=self.random_state,
                max_iter=1000,
            )
        elif self.classifier_type == "svm":
            clf = SVC(
                probability=True,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )
        else:  # decision_tree
            clf = DecisionTreeClassifier(
                max_depth=6,
                class_weight=self.class_weight,
                random_state=self.random_state,
            )

        clf.fit(X, y)
        self.classifier = clf

    def is_trained(self) -> bool:
        """Return True if ``fit`` has been called successfully."""
        return self.classifier is not None

    # ------------------------------------------------------------------
    # TerminationStrategy interface
    # ------------------------------------------------------------------

    def should_terminate(self, node: Node, path: List[Node]) -> bool:
        depth = node.get("depth", len(path) - 1)

        # Depth fallback (always applied)
        if depth >= self.max_depth_fallback:
            return True

        # Untrained: no learned termination, rely on other strategies
        if self.classifier is None:
            return False

        P_log = node.get("P")
        if P_log is None:
            return False

        # exp(log_P) clipped to [0, 1]
        p_val = min(1.0, max(0.0, math.exp(P_log)))
        X = np.array([[p_val]], dtype=np.float64)

        # classes_ order: sklearn guarantees sorted → [0, 1]
        classes = list(self.classifier.classes_)
        prob_terminate = float(self.classifier.predict_proba(X)[0][classes.index(1)])

        if self.use_stochastic:
            return self._rng.random() < prob_terminate
        return prob_terminate >= 0.5

    # ------------------------------------------------------------------
    # Data collection helper (standalone, no vLLM dependency at import)
    # ------------------------------------------------------------------

    @staticmethod
    def collect_training_data(
        server_url: str,
        model_name: str,
        dataset: List[Dict[str, Any]],
        gold_key: str = "answer",
        prompt_key: str = "prompt",
        n_samples: int = 3,
        max_tokens: int = 256,
        temperature: float = 0.8,
        top_p: float = 1.0,
        seed: Optional[int] = None,
    ) -> "Tuple[List[float], List[int]]":
        """Sample trajectories from a small dataset and build (P, label) pairs.

        For each problem in *dataset*, ``n_samples`` continuations are sampled
        from vLLM; P(answer|trajectory) is computed for each; the label is
        ``1`` (terminate) if P fell relative to the root, ``0`` otherwise.

        Parameters
        ----------
        server_url : str
            vLLM API base URL.
        model_name : str
            Model identifier on the vLLM server.
        dataset : list of dict
            Each element must have keys *prompt_key* and *gold_key*.
        gold_key : str
            Key for the gold answer in each dataset record.
        prompt_key : str
            Key for the prompt in each dataset record.
        n_samples : int
            Number of continuations to sample per problem.
        max_tokens, temperature, top_p, seed
            Generation parameters forwarded to vLLM.

        Returns
        -------
        P_values : list of float
            exp(log_P) values for each sampled node.
        labels : list of int
            1 = terminate (P degraded), 0 = continue.
        """
        from mc_analysis.vllm_helpers import (
            _make_session,
            _sample_completions,
            compute_sequence_logprob,
        )

        session = _make_session(api_key=None)
        P_values: List[float] = []
        labels: List[int] = []

        for record in dataset:
            prompt = record[prompt_key]
            gold = record[gold_key]

            root_result = compute_sequence_logprob(server_url, model_name, prompt, gold)
            P_root = float(root_result["sum_logprob"])

            children = _sample_completions(
                server_url=server_url,
                model_name=model_name,
                prefix=prompt,
                n=n_samples,
                max_tokens=max_tokens,
                temperature=temperature,
                top_p=top_p,
                stop=None,
                get_logprobs=False,
                seed=seed,
                session=session,
            )

            for child in children:
                full_text = prompt + child.get("text", "")
                child_result = compute_sequence_logprob(
                    server_url, model_name, full_text, gold
                )
                P_child = float(child_result["sum_logprob"])
                p_val = min(1.0, max(0.0, math.exp(P_child)))
                P_values.append(p_val)
                labels.append(1 if P_child < P_root else 0)

        return P_values, labels
