"""
Answer extraction and equivalence checking.
Re-uses adpo.reward_functions when available (LaTeX, numeric, fraction, …).
"""
import re
from typing import Optional


def _try_import():
    try:
        from adpo.reward_functions import extract_boxed_answer, is_equiv
        return extract_boxed_answer, is_equiv
    except ImportError:
        return None, None


_ext_boxed, _is_equiv = _try_import()


def extract_answer(text: str) -> Optional[str]:
    """Extract final answer: tries \\boxed{…} first, falls back to last line."""
    if _ext_boxed is not None:
        ans = _ext_boxed(text)
        if ans is not None:
            return ans
    m = re.search(r"\\boxed\{([^{}]*)\}", text)
    if m:
        return m.group(1).strip()
    lines = [ln.strip() for ln in text.splitlines() if ln.strip()]
    return lines[-1] if lines else None


def default_answer_checker(generated: Optional[str], gold: str) -> bool:
    """Equivalence check (LaTeX-aware when adpo.reward_functions is available)."""
    gen = (generated or "").strip()
    ref = (gold or "").strip()
    if _is_equiv is not None:
        return bool(_is_equiv(gen, ref))
    if gen.lower() == ref.lower():
        return True
    try:
        return abs(float(gen) - float(ref)) < 1e-5
    except (ValueError, TypeError):
        pass
    try:
        import sympy
        return bool(sympy.simplify(sympy.sympify(gen) - sympy.sympify(ref)) == 0)
    except Exception:
        pass
    return False
