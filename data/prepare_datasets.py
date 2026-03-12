"""
Data preparation script for ADPO training and evaluation.

Converts HuggingFace datasets into verl's required parquet format with
five fields: data_source, prompt, ability, reward_model, extra_info.

Usage:
    python data/prepare_datasets.py --dataset gsm8k --output_dir data/processed/
    python data/prepare_datasets.py --dataset all --output_dir data/processed/
"""

import argparse
import json
import os

import datasets
import pandas as pd


SYSTEM_PROMPT = "Please reason step by step, and put your final answer within \\boxed{}."


def make_prompt(question: str) -> list:
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user", "content": question},
    ]


# ---------------------------------------------------------------------------
# Training dataset loaders
# ---------------------------------------------------------------------------

def load_gsm8k(split="train"):
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


def load_math(split="train"):
    ds = datasets.load_dataset("hendrycks/competition_math", split=split)
    records = []
    for i, row in enumerate(ds):
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


def load_math500():
    ds = datasets.load_dataset("HuggingFaceH4/MATH-500", split="test")
    records = []
    for i, row in enumerate(ds):
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


def load_numina_math(split="train"):
    ds = datasets.load_dataset("AI-MO/NuminaMath-1.5", split=split)
    records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        answer = row.get("solution", row.get("answer", ""))
        records.append({
            "data_source": "numina_math",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i, "source": row.get("source", "")},
        })
    return records


def load_open_math_reasoning(split="train"):
    ds = datasets.load_dataset("nvidia/OpenMathReasoning", split=split, streaming=True)
    records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        answer = row.get("expected_answer", row.get("solution", ""))
        records.append({
            "data_source": "open_math_reasoning",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i},
        })
        if i >= 500_000:
            break
    return records


def load_aops_instruct(split="train"):
    ds = datasets.load_dataset("qq8933/AoPS-Instruct", split=split)
    records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        answer = row.get("solution", row.get("answer", ""))
        records.append({
            "data_source": "aops_instruct",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i},
        })
    return records


def load_big_math_rl(split="train"):
    ds = datasets.load_dataset("SynthLabsAI/Big-Math-RL-Verified", split=split)
    records = []
    for i, row in enumerate(ds):
        question = row.get("problem", row.get("question", ""))
        answer = row.get("solution", row.get("answer", ""))
        records.append({
            "data_source": "big_math_rl",
            "prompt": make_prompt(question),
            "ability": "math",
            "reward_model": {"ground_truth": answer},
            "extra_info": {"split": split, "index": i},
        })
    return records


def load_fine_math(split="train"):
    ds = datasets.load_dataset("HuggingFaceTB/finemath", split=split)
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "fine_math",
            "prompt": make_prompt(row.get("text", "")),
            "ability": "math",
            "reward_model": {"ground_truth": ""},
            "extra_info": {"split": split, "index": i},
        })
        if i >= 200_000:
            break
    return records


# ---------------------------------------------------------------------------
# Evaluation dataset loaders
# ---------------------------------------------------------------------------

def load_amc_2023():
    ds = datasets.load_dataset("AI-MO/amc-aime-dataset", split="test")
    records = []
    for i, row in enumerate(ds):
        year = str(row.get("year", ""))
        comp = row.get("competition", "").lower()
        if "2023" in year and "amc" in comp:
            records.append({
                "data_source": "amc_2023",
                "prompt": make_prompt(row["problem"]),
                "ability": "math",
                "reward_model": {"ground_truth": str(row["answer"])},
                "extra_info": {"split": "test", "index": i, "year": year},
            })
    return records


def load_aime_2024():
    ds = datasets.load_dataset("AI-MO/amc-aime-dataset", split="test")
    records = []
    for i, row in enumerate(ds):
        year = str(row.get("year", ""))
        comp = row.get("competition", "").lower()
        if "2024" in year and "aime" in comp:
            records.append({
                "data_source": "aime_2024",
                "prompt": make_prompt(row["problem"]),
                "ability": "math",
                "reward_model": {"ground_truth": str(row["answer"])},
                "extra_info": {"split": "test", "index": i},
            })
    return records


def load_aime_2025():
    ds = datasets.load_dataset("opencompass/AIME2025", split="test")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "aime_2025",
            "prompt": make_prompt(row.get("problem", row.get("question", ""))),
            "ability": "math",
            "reward_model": {"ground_truth": str(row.get("answer", row.get("solution", "")))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_olympiad_bench():
    ds = datasets.load_dataset("lmms-lab/OlympiadBench", split="test_en")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "olympiad_bench",
            "prompt": make_prompt(row.get("question", "")),
            "ability": "math",
            "reward_model": {"ground_truth": row.get("final_answer", [""])[0]},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_minerva_math():
    ds = datasets.load_dataset("math-ai/minerva_math", split="test")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "minerva_math",
            "prompt": make_prompt(row.get("problem", "")),
            "ability": "math",
            "reward_model": {"ground_truth": row.get("solution", "")},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_omni_math():
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


def load_hmmt():
    ds = datasets.load_dataset("keirp/HMMT", split="test")
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


def load_brumo():
    ds = datasets.load_dataset("Idavidrein/BRUMO", split="test")
    records = []
    for i, row in enumerate(ds):
        records.append({
            "data_source": "brumo",
            "prompt": make_prompt(row.get("question", row.get("problem", ""))),
            "ability": "math",
            "reward_model": {"ground_truth": str(row.get("answer", row.get("solution", "")))},
            "extra_info": {"split": "test", "index": i},
        })
    return records


def load_cmimc():
    ds = datasets.load_dataset("I-Manuella/CMIMC_math", split="test")
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
    "math500": load_math500, "gsm8k_test": lambda: load_gsm8k(split="test"),
    "amc_2023": load_amc_2023, "aime_2024": load_aime_2024, "aime_2025": load_aime_2025,
    "olympiad_bench": load_olympiad_bench, "minerva_math": load_minerva_math,
    "omni_math": load_omni_math, "hmmt": load_hmmt, "brumo": load_brumo, "cmimc": load_cmimc,
}


def save_to_parquet(records, output_path):
    rows = []
    for r in records:
        rows.append({
            "data_source": r["data_source"],
            "prompt": json.dumps(r["prompt"], ensure_ascii=False),
            "ability": r["ability"],
            "reward_model": json.dumps(r["reward_model"], ensure_ascii=False),
            "extra_info": json.dumps(r["extra_info"], ensure_ascii=False),
        })
    df = pd.DataFrame(rows)
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    df.to_parquet(output_path, index=False)
    print(f"Saved {len(df)} examples to {output_path}")


def main():
    parser = argparse.ArgumentParser(description="Prepare datasets for ADPO")
    parser.add_argument("--dataset", type=str, required=True)
    parser.add_argument("--split", type=str, default="train")
    parser.add_argument("--output_dir", type=str, default="data/processed")
    args = parser.parse_args()

    if args.dataset in ("all_train", "all"):
        for name, loader in TRAIN_DATASETS.items():
            print(f"\n--- Processing {name} ---")
            try:
                recs = loader(split=args.split) if "split" in loader.__code__.co_varnames else loader()
                save_to_parquet(recs, os.path.join(args.output_dir, "train", f"{name}.parquet"))
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    if args.dataset in ("all_eval", "all"):
        for name, loader in EVAL_DATASETS.items():
            print(f"\n--- Processing {name} ---")
            try:
                recs = loader()
                save_to_parquet(recs, os.path.join(args.output_dir, "eval", f"{name}.parquet"))
            except Exception as e:
                print(f"  [SKIP] {name}: {e}")

    if args.dataset not in ("all_train", "all_eval", "all"):
        all_ds = {**TRAIN_DATASETS, **EVAL_DATASETS}
        if args.dataset not in all_ds:
            print(f"Unknown dataset: {args.dataset}\nAvailable: {list(all_ds.keys())}")
            return
        loader = all_ds[args.dataset]
        try:
            recs = loader(split=args.split) if "split" in loader.__code__.co_varnames else loader()
        except TypeError:
            recs = loader()
        subset = "eval" if args.dataset in EVAL_DATASETS else "train"
        save_to_parquet(recs, os.path.join(args.output_dir, subset, f"{args.dataset}.parquet"))


if __name__ == "__main__":
    main()
