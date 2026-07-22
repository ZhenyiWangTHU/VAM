"""Command-line interface for reference, score, and aggregate DNB stages."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from .aggregate import aggregate_idnb, aggregate_stages
from .reference import build_reference_network
from .score import DNBConfig, score_cohort


def build_reference(args: argparse.Namespace) -> None:
    payload = np.load(args.input, allow_pickle=False)
    correlation, mask, edges = build_reference_network(
        payload["embeddings"], payload["feature_names"].astype(str).tolist(), args.pvalue
    )
    np.savez_compressed(
        args.output,
        reference_samples=payload["embeddings"],
        feature_names=payload["feature_names"],
        correlation=correlation,
        mask=mask,
    )
    edges.to_csv(Path(args.output).with_suffix(".edges.csv"), index=False)


def score(args: argparse.Namespace) -> None:
    reference = np.load(args.reference, allow_pickle=False)
    cohort = np.load(args.input, allow_pickle=False)
    config = DNBConfig(
        ssn_pvalue=args.pvalue,
        min_hub_degree=args.min_hub_degree,
        two_sided=args.two_sided,
    )
    sample_ids = cohort["sample_ids"].astype(str)
    embeddings = cohort["embeddings"]
    stages = cohort["stages"] if "stages" in cohort.files else None
    existing = None
    if args.resume and Path(args.output).exists():
        existing = pd.read_csv(args.output, dtype={"sample_id": str})
        completed = set(existing["sample_id"].unique())
        pending = np.array([sample_id not in completed for sample_id in sample_ids])
        sample_ids, embeddings = sample_ids[pending], embeddings[pending]
        stages = stages[pending] if stages is not None else None
    result = score_cohort(
        reference["reference_samples"],
        embeddings,
        reference["correlation"],
        reference["mask"],
        sample_ids.tolist(),
        stages,
        reference["feature_names"].astype(str).tolist(),
        config,
        args.n_jobs,
    )
    if existing is not None:
        result = pd.concat([existing, result], ignore_index=True)
    result.to_csv(args.output, index=False)


def aggregate(args: argparse.Namespace) -> None:
    scores = pd.read_csv(args.input, dtype={"sample_id": str})
    idnb = aggregate_idnb(scores, args.top_k)
    idnb.to_csv(args.output, index=False)
    if "stage" in idnb:
        aggregate_stages(idnb, args.stage_method).to_csv(
            Path(args.output).with_suffix(".stages.csv"), index=False
        )


def main() -> None:
    parser = argparse.ArgumentParser(prog="vam-dnb")
    subparsers = parser.add_subparsers(required=True)
    reference_parser = subparsers.add_parser("build-reference")
    reference_parser.add_argument("--input", required=True)
    reference_parser.add_argument("--output", required=True)
    reference_parser.add_argument("--pvalue", type=float, default=0.01)
    reference_parser.set_defaults(function=build_reference)

    score_parser = subparsers.add_parser("score")
    score_parser.add_argument("--reference", required=True)
    score_parser.add_argument("--input", required=True)
    score_parser.add_argument("--output", required=True)
    score_parser.add_argument("--pvalue", type=float, default=0.25)
    score_parser.add_argument("--min-hub-degree", type=int, default=3)
    score_parser.add_argument("--two-sided", action="store_true")
    score_parser.add_argument("--n-jobs", type=int, default=1)
    score_parser.add_argument("--resume", action="store_true")
    score_parser.set_defaults(function=score)

    aggregate_parser = subparsers.add_parser("aggregate")
    aggregate_parser.add_argument("--input", required=True)
    aggregate_parser.add_argument("--output", required=True)
    aggregate_parser.add_argument("--top-k", type=int, default=5)
    aggregate_parser.add_argument("--stage-method", choices=["max", "mean", "median"], default="max")
    aggregate_parser.set_defaults(function=aggregate)
    args = parser.parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
