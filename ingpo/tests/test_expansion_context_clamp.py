import sys
from pathlib import Path
import types


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "spo" / "src"))
if "datasets" not in sys.modules:
    datasets_stub = types.ModuleType("datasets")
    datasets_stub.Dataset = object
    sys.modules["datasets"] = datasets_stub
if "overrides" not in sys.modules:
    overrides_stub = types.ModuleType("overrides")
    overrides_stub.overrides = lambda f=None, *args, **kwargs: f
    sys.modules["overrides"] = overrides_stub

from treetune.inference_strategies.tree_inference.expansion import EfficientIIDExpander


class _DummyTokenizer:
    def __init__(self, count: int):
        self._count = count

    def tokenize(self, _: str):
        return ["x"] * self._count


class _TestExpander(EfficientIIDExpander):
    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self.called = False

    async def _run_program(self, program, prefix):
        self.called = True
        raise AssertionError("_run_program should not be called when max_tokens <= 0")


def test_sample_node_returns_truncated_leaf_when_context_exhausted():
    expander = _TestExpander(
        branch_factor_strategy=lambda _: 2,
        num_expansion_rounds=1,
        program='{{prefix}}{{gen "chain_of_thought" max_tokens={max_tokens} n={num_samples}}}',
        program_kwargs={"max_tokens": 4096},
        node_text_template="{chain_of_thought}",
        model_context_size=100,
        tokenizer=_DummyTokenizer(count=100),
    )

    import asyncio

    nodes = asyncio.run(
        expander._sample_node(
            prefix="short",
            depth=1,
            branch_factor=2,
            max_tokens=600,
        )
    )

    assert expander.called is False
    assert len(nodes) == 2
    assert all(node["finish_reason"] == "length" for node in nodes)
    assert all(node["full_text"] == "short" for node in nodes)
