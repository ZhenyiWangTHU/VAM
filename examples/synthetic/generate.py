"""Generate a tiny, non-biological dataset for CLI smoke tests."""

from pathlib import Path

import numpy as np
import pandas as pd


output = Path(__file__).parent
rng = np.random.default_rng(7)
sample_ids = [f"sample_{idx:03d}" for idx in range(40)]

sequence = pd.DataFrame(
    rng.normal(size=(40, 8)), index=sample_ids, columns=[f"esm2_{idx}" for idx in range(8)]
)
expression = pd.DataFrame(
    rng.normal(size=(40, 6)), index=sample_ids, columns=[f"scgpt_{idx}" for idx in range(6)]
)
metabolite = pd.DataFrame(
    rng.lognormal(size=(40, 5)), index=sample_ids, columns=[f"metabolite_{idx}" for idx in range(5)]
)
age = 50 + sequence.iloc[:, 0] * 2 + expression.iloc[:, 0] + metabolite.iloc[:, 0] * 0.2

for name, frame in {
    "sequence.tsv": sequence,
    "expression.tsv": expression,
    "metabolite.tsv": metabolite,
    "labels.tsv": age.rename("age").to_frame(),
}.items():
    frame.to_csv(output / name, sep="\t", index_label="sample_id")
(output / "validation_ids.txt").write_text("\n".join(sample_ids[-8:]) + "\n", encoding="utf-8")
