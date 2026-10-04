#!/usr/bin/env python3
"""Download openai/gsm8k and write data/gsm8k/{train,test}.parquet in the format data_handlers/gsm8k.py reads.

Same format as baselines/examples/data_preprocess/gsm8k.py, but without the verl dependency:
  python scripts/prepare_gsm8k.py [--out_dir data/gsm8k]
"""
import argparse
import os
import re

import pandas as pd

INSTRUCTION = 'Let\'s think step by step and output the final answer after "####".'


def final_answer(answer: str) -> str:
    m = re.search(r"#### (\-?[0-9\.\,]+)", answer)
    assert m is not None, answer
    return m.group(1).replace(",", "")


def main():
    p = argparse.ArgumentParser()
    p.add_argument("--out_dir", default="data/gsm8k")
    args = p.parse_args()

    import datasets
    dataset = datasets.load_dataset("openai/gsm8k", "main")
    os.makedirs(args.out_dir, exist_ok=True)
    for split in ("train", "test"):
        rows = []
        for idx, ex in enumerate(dataset[split]):
            rows.append({
                "data_source": "openai/gsm8k",
                "prompt": [{"role": "user", "content": ex["question"] + " " + INSTRUCTION}],
                "ability": "math",
                "reward_model": {"style": "rule", "ground_truth": final_answer(ex["answer"])},
                "extra_info": {"split": split, "index": idx, "answer": ex["answer"], "question": ex["question"]},
            })
        path = os.path.join(args.out_dir, f"{split}.parquet")
        pd.DataFrame(rows).to_parquet(path)
        print(f"{path}: {len(rows)} rows")


if __name__ == "__main__":
    main()
