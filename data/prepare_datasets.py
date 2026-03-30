"""
Data preparation script for ADPO training and evaluation.

Converts HuggingFace datasets into verl's required parquet format with
five fields: data_source, prompt, ability, reward_model, extra_info.

Usage:
    python data/prepare_datasets.py --dataset gsm8k --output_dir data/processed/
    python data/prepare_datasets.py --dataset all --output_dir data/processed/
    python data/prepare_datasets.py --dataset math --level "Level 4,Level 5" --output_dir data/processed/
    python data/prepare_datasets.py --dataset numina_math --level "amc_aime,olympiads" --output_dir data/processed/
"""

import argparse
import gc
import os

import datasets
import pandas as pd

BATCH_SIZE = 10_000  # write parquet every N records to avoid OOM


SYSTEM_PROMPT = "Please reason step by step, each reasoning phase should ended with '.', and put your final answer within \\boxed{}."


def make_prompt(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Level filtering support
# ---------------------------------------------------------------------------
# Each dataset can define a "level_field" — the key in the HF row used for
# difficulty / category filtering.  When --level is passed on the CLI the
# value(s) are matched against that field.
#
# Dataset          level_field     Example values
# -------          -----------     --------------
# math / math500   level           "Level 1" … "Level 5"
# numina_math      source          "amc_aime", "olympiads", "cn_k12", …
# open_math_rsng   problem_source  "MATH", "aops", "AMC", …
# big_math_rl      source          (varies)
# fine_math        source          "cn_k12", "orca_math", "synthetic_math"
# aops_instruct    problem_source  "aops_forum", …

DATASET_LEVEL_FIELD = {
    "math": "level",
    "math500": "level",
    "numina_math": "source",
    "open_math_reasoning": "problem_source",
    "aops_instruct": "problem_source",
    "big_math_rl": "source",
    "fine_math": "source",
}


def _parse_levels(level_str):
    """Parse comma-separated level string into a set of lowercase values."""
    if not level_str:
        return None
    return {v.strip().lower() for v in level_str.split(",") if v.strip()}


def _match_level(row_value, allowed_levels):
    """Check if a row's level value matches any of the allowed levels."""
    if allowed_levels is None:
        return True
    return str(row_value).strip().lower() in allowed_levels


# ---------------------------------------------------------------------------
# Training dataset loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split="train", levels=None):
    ds = datasets.load_dataset("openai/gsm8k", "main", split=split)
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "gsm8k",
            "prompt": make_prompt(row["question"]),
            "ability": "math",
            "reward_model": {"ground_truth": row["answer"]},
            "extra_info": {"split": split, "index": i},
        })
    return records


def load_math(split="train", levels=None):
    try:
        ds = datasets.load_dataset("lighteval/MATH", split=split)
    except Exception:
        ds = datasets.load_dataset("DigitalLearningGmbH/MATH-lighteval", split=split)
    records = []
    for i, row in enumerate(ds):
        if levels and not _match_level(row.get("level", ""), levels):
            continue
        records.append({
            "data_source": "math",
            "prompt": make_prompt(row["problem"]),
            "ability": "math",
            "reward_model": {"ground_truth": row["solution"]},
            "extra_info": {
                "split": split, "index": i,
                "level": row.get("level", ""), "type": row.get("type", ""),
            },
        })
    return records


def load_math500(levels=None):
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = []
    for i, row in enumerate(ds):
        if levels and not _match_level(row.get("level", ""), levels):
            continue
        records.append({
            "data_source": "math500",
            "prompt": make_prompt(row["problem"]),
            "ability": "math",
            "reward_model": {"ground_truth": row["answer"]},
            "extra_info": {
                "split": "test", "index": i,
                "level": row.get("level", ""), "type": row.get("subject", ""),
            },
        })
    return records


def load_numina_math(split="train", levels=None):
    ds = datasets.load_dataset("AI-MO/NuminaMath-1.5", split=split, streaming=True)
    for i, row in enumerate(ds):
        if levels and not _match_level(row.get("source", ""), levels):
            continue
        question = row.get("problem", row.get("question", ""))
        answer = row.get("solution", row.get("answer", ""))
        yield {
            "data_source": "numina_math",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "source": row.get("source", "")},
        }


def load_open_math_reasoning(split="cot", levels=None):
    ds = datasets.load_dataset("nvidia/OpenMathReasoning", split=split, streaming=True)
    for i, row in enumerate(ds):
        if levels and not _match_level(row.get("problem_source", ""), levels):
            continue
        question = row.get("problem", row.get("question", ""))
        answer = row.get("expected_answer", row.get("solution", ""))
        yield {
            "data_source": "open_math_reasoning",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "problem_source": row.get("problem_source", "")},
        }
        if i >= 500_000:
            break


def load_aops_instruct(split="train", levels=None):
    """AoPS problems via OpenMathReasoning (AoPS-sourced subset)."""
    ds = datasets.load_dataset("nvidia/OpenMathReasoning", split="cot", streaming=True)
    count = 0
    for i, row in enumerate(ds):
        source = row.get("problem_source", "")
        if "aops" not in source.lower():
            continue
        if levels and not _match_level(source, levels):
            continue
        question = row.get("problem", row.get("question", ""))
        answer = row.get("expected_answer", row.get("solution", ""))
        yield {
            "data_source": "aops_instruct",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "source": source},
        }
        count += 1
        if count >= 100_000:
            break


def load_big_math_rl(split="train", levels=None):
    try:
        ds = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified", split=split, streaming=True)
    except Exception:
        ds = datasets.load_dataset("open-r1/Big-Math-RL-Verified-Processed", "all", split=split, streaming=True)
    for i, row in enumerate(ds):
        if levels and not _match_level(row.get("source", ""), levels):
            continue
        question = row.get("problem", row.get("question", ""))
        answer = row.get("answer", row.get("solution", ""))
        yield {
            "data_source": "big_math_rl",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "source": row.get("source", "")},
        }


def load_fine_math(split="train", levels=None):
    """FineMath is a web corpus (no Q&A pairs). Use NuminaMath-1.5 filtered subset instead."""
    ds = datasets.load_dataset("AI-MO/NuminaMath-1.5", split=split, streaming=True)
    count = 0
    for i, row in enumerate(ds):
        source = row.get("source", "")
        if source not in ("cn_k12", "orca_math", "synthetic_math"):
            continue
        if levels and not _match_level(source, levels):
            continue
        question = row.get("problem", row.get("question", ""))
        answer = row.get("solution", row.get("answer", ""))
        yield {
            "data_source": "fine_math",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "source": source},
        }
        count += 1
        if count >= 200_000:
            break


# ---------------------------------------------------------------------------
# Evaluation dataset loaders
# ---------------------------------------------------------------------------

def load_amc_2023(levels=None):
    ds = datasets.load_dataset("AI-MO/aimo-validation-amc", split="train")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "amc_2023",
            "prompt": make_prompt(row["problem"]),
            "ability": "math",
            "reward_model": {"ground_truth": str(row["answer"])},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_aime_2024(levels=None):
    ds = datasets.load_dataset("AI-MO/aimo-validation-aime", split="train")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "aime_2024",
            "prompt": make_prompt(row["problem"]),
            "ability": "math",
            "reward_model": {"ground_truth": str(row.get("answer", row.get("solution", "")))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_aime_2025(levels=None):
    records = []
    for config in ("AIME2025-I", "AIME2025-II"):
        ds = datasets.load_dataset("opencompass/AIME2025", config, split="test")
        for i, row in enumerate(ds):
            records.append({
                "data_source": "aime_2025",
                "prompt": make_prompt(row.get("question", row.get("problem", ""))),
                "ability": "math",
                "reward_model": {"ground_truth": str(row.get("answer", ""))},
                "extra_info": {"split": "test", "index": len(records), "config": config},
            })
    return records


def load_olympiad_bench(levels=None):
    ds = datasets.load_dataset("lmms-lab/OlympiadBench", split="test_en")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "olympiad_bench",
            "prompt": make_prompt(row.get("question", "")),
            "ability": "math",
            "reward_model": {"ground_truth": (row.get("final_answer") or [""])[0]},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_minerva_math(levels=None):
    ds = datasets.load_dataset("math-ai/minervamath", split="test")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "minerva_math",
            "prompt": make_prompt(row.get("question", row.get("problem", ""))),
            "ability": "math",
            "reward_model": {"ground_truth": row.get("answer", row.get("solution", ""))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_omni_math(levels=None):
    ds = datasets.load_dataset("KbsdJames/Omni-MATH", split="test")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "omni_math",
            "prompt": make_prompt(row.get("problem", "")),
            "ability": "math",
            "reward_model": {"ground_truth": row.get("solution", "")},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_hmmt(levels=None):
    ds = datasets.load_dataset("MathArena/hmmt_feb_2025", split="train")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "hmmt",
            "prompt": make_prompt(row.get("problem", "")),
            "ability": "math",
            "reward_model": {"ground_truth": row.get("answer", "")},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_brumo(levels=None):
    ds = datasets.load_dataset("MathArena/brumo_2025", split="train")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "brumo",
            "prompt": make_prompt(row.get("problem", row.get("question", ""))),
            "ability": "math",
            "reward_model": {"ground_truth": str(row.get("answer", ""))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_cmimc(levels=None):
    ds = datasets.load_dataset("MathArena/cmimc_2025", split="train")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "cmimc",
            "prompt": make_prompt(row.get("problem", "")),
            "ability": "math",
            "reward_model": {"ground_truth": str(row.get("answer", ""))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


# ---------------------------------------------------------------------------
# Registry & main
# ---------------------------------------------------------------------------

TRAIN_DATASETS = {
    "gsm8k": load_gsm8k, "math": load_math, "numina_math": load_numina_math,
    "open_math_reasoning": load_open_math_reasoning, "aops_instruct": load_aops_instruct,
    "big_math_rl": load_big_math_rl, "fine_math": load_fine_math,
}

EVAL_DATASETS = {
    "math500": load_math500, "gsm8k_test": lambda levels=None: load_gsm8k(split="test", levels=levels),
    "amc_2023": load_amc_2023, "aime_2024": load_aime_2024, "aime_2025": load_aime_2025,
    "olympiad_bench": load_olympiad_bench, "minerva_math": load_minerva_math,
    "omni_math": load_omni_math, "hmmt": load_hmmt, "brumo": load_brumo, "cmimc": load_cmimc,
}


def save_to_parquet(records, output_path):
    """Save records to parquet with native types (dict/list columns)."""
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    all_records = []
    for i, r in enumerate(records):
        all_records.append({
            "data_source": r["data_source"],
            "prompt": r["prompt"],       # list of dicts — keep native
            "ability": r["ability"],
            "reward_model": r["reward_model"],  # dict — keep native
            "extra_info": r["extra_info"],      # dict — keep native
        })
        if (i + 1) % BATCH_SIZE == 0:
            gc.collect()

    df = pd.DataFrame(all_records)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} examples to {output_path}")


def _call_loader(loader, split="train", levels=None):
    """Call a loader with the appropriate kwargs it supports."""
    import inspect
    sig = inspect.signature(loader)
    kwargs = {}
    if "split" in sig.parameters:
        kwargs["split"] = split
    if "levels" in sig.parameters:
        kwargs["levels"] = levels
    return loader(**kwargs)


def _output_name(dataset_name, levels):
    """Generate output filename, appending level suffix if filtering."""
    if levels:
        suffix = "_".join(sorted(levels))
        return f"{dataset_name}_{suffix}"
    return dataset_name


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ADPO")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    parser.add_argument(
        "--level", type=str, default=None,
        help=(
            "Comma-separated level/category filter. The field used depends on the dataset: "
            "MATH/MATH500 filter by 'level' (e.g. 'Level 4,Level 5'), "
            "NuminaMath by 'source' (e.g. 'amc_aime,olympiads'), "
            "OpenMathReasoning by 'problem_source'. "
            "Ignored for datasets without level info."
        ),
    )
    args = parser.parse_args()
    levels = _parse_levels(args.level)

    if levels:
        ds_name = args.dataset if args.dataset not in ("all_train", "all_eval", "all") else "multiple"
        level_field = DATASET_LEVEL_FIELD.get(ds_name, "?")
        print(f"Level filter active: {args.level} (field: {level_field})")

    if args.dataset in ("all_train", "all"):
        for name, loader in TRAIN_DATASETS.items():
            print(f"\n--- Processing {name} ---")
            try:
                recs = _call_loader(loader, split=args.split, levels=levels)
                out_name = _output_name(name, levels)
                save_to_parquet(recs, os.path.join(args.output_dir, "train", f"{out_name}.parquet"))
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    if args.dataset in ("all_eval", "all"):
        for name, loader in EVAL_DATASETS.items():
            print(f"\n--- Processing {name} ---")
            try:
                recs = _call_loader(loader, levels=levels)
                out_name = _output_name(name, levels)
                save_to_parquet(recs, os.path.join(args.output_dir, "eval", f"{out_name}.parquet"))
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    if args.dataset not in ("all_train", "all_eval", "all"):
        all_ds = {**TRAIN_DATASETS, **EVAL_DATASETS}
        if args.dataset not in all_ds:
            print(f"Unknown dataset: {args.dataset}\nAvailable: {list(all_ds.keys())}")
            return
        loader = all_ds[args.dataset]
        recs = _call_loader(loader, split=args.split, levels=levels)
        subset = "eval" if args.dataset in EVAL_DATASETS else "train"
        out_name = _output_name(args.dataset, levels)
        save_to_parquet(recs, os.path.join(args.output_dir, subset, f"{out_name}.parquet"))


if __name__ == "__main__":
    main()
