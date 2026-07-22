"""Command-line entry points for VAM training and inference."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import yaml

from .checkpoint import load_checkpoint
from .data.alignment import align_samples
from .data.metabolite import normalize_metabolite_abundance
from .data.schema import FeatureSchema
from .inference import predict
from .training import TrainingConfig, fit


def _table(path: str) -> pd.DataFrame:
    separator = "\t" if Path(path).suffix.lower() in {".tsv", ".txt"} else ","
    frame = pd.read_csv(path, sep=separator, index_col=0)
    frame.index = frame.index.map(str)
    return frame


def _aligned(args: argparse.Namespace, with_labels: bool):
    sequence, expression, metabolite = (
        _table(args.sequence),
        _table(args.expression),
        _table(args.metabolite),
    )
    labels = None
    if with_labels:
        label_table = _table(args.labels)
        labels = label_table[args.age_column]
    return align_samples(sequence, expression, metabolite, labels)


def train_command(args: argparse.Namespace) -> None:
    sequence, expression, metabolite, labels = _aligned(args, True)
    assert labels is not None
    validation_ids = {
        line.strip() for line in Path(args.validation_ids).read_text(encoding="utf-8").splitlines()
        if line.strip()
    }
    validation_mask = sequence.index.isin(validation_ids)
    if not validation_mask.any() or validation_mask.all():
        raise ValueError("Validation IDs must define a non-empty proper subset")
    training_metabolite = metabolite.loc[~validation_mask].apply(
        pd.to_numeric, errors="raise"
    )
    metabolite_target_sum = float(training_metabolite.sum(axis=1).mean())
    metabolite = normalize_metabolite_abundance(
        metabolite, target_sum=metabolite_target_sum
    )
    config_data = yaml.safe_load(Path(args.config).read_text(encoding="utf-8")) or {}
    schema = FeatureSchema(
        sequence_feature_names=sequence.columns.astype(str).tolist(),
        expression_feature_names=expression.columns.astype(str).tolist(),
        metabolite_names=metabolite.columns.astype(str).tolist(),
        embedding_dim=int(config_data.pop("embedding_dim", 256)),
        metabolite_target_sum=metabolite_target_sum,
    )
    arrays = tuple(frame.to_numpy(np.float32) for frame in (sequence, expression, metabolite))
    metrics = fit(
        tuple(item[~validation_mask] for item in arrays),
        labels.to_numpy(np.float32)[~validation_mask],
        tuple(item[validation_mask] for item in arrays),
        labels.to_numpy(np.float32)[validation_mask],
        schema,
        args.checkpoint,
        TrainingConfig(**config_data),
        args.device,
    )
    print(json.dumps(metrics, indent=2))


def infer_command(args: argparse.Namespace) -> None:
    sequence, expression, metabolite, _ = _aligned(args, False)
    _, schema, _ = load_checkpoint(args.checkpoint, "cpu")
    sequence_names = schema.sequence_feature_names
    expression_names = schema.expression_feature_names
    for label, frame, names in (
        ("sequence", sequence, sequence_names),
        ("expression", expression, expression_names),
        ("metabolite", metabolite, schema.metabolite_names),
    ):
        missing = pd.Index(names).difference(frame.columns)
        extra = frame.columns.difference(names)
        if len(missing) or len(extra):
            raise ValueError(
                f"{label} feature set mismatch; missing={missing[:5].tolist()}, "
                f"extra={extra[:5].tolist()}"
            )
    sequence = sequence.loc[:, sequence_names]
    expression = expression.loc[:, expression_names]
    metabolite = metabolite.loc[:, schema.metabolite_names]
    schema.validate_feature_names(
        sequence.columns.astype(str).tolist(),
        expression.columns.astype(str).tolist(),
        metabolite.columns.astype(str).tolist(),
    )
    metabolite = normalize_metabolite_abundance(
        metabolite,
        feature_names=schema.metabolite_names,
        target_sum=schema.metabolite_target_sum,
    )
    ages, embeddings = predict(
        args.checkpoint,
        tuple(frame.to_numpy(np.float32) for frame in (sequence, expression, metabolite)),
        args.batch_size,
        args.device,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        args.output,
        sample_ids=sequence.index.to_numpy(str),
        predicted_age=ages,
        embeddings=embeddings,
    )


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(prog="vam")
    subparsers = parser.add_subparsers(required=True)
    for name in ("train", "infer"):
        command = subparsers.add_parser(name)
        command.add_argument("--sequence", required=True)
        command.add_argument("--expression", required=True)
        command.add_argument("--metabolite", required=True)
        command.add_argument("--device", default=None)
        if name == "train":
            command.add_argument("--labels", required=True)
            command.add_argument("--age-column", default="age")
            command.add_argument("--validation-ids", required=True)
            command.add_argument("--config", required=True)
            command.add_argument("--checkpoint", required=True)
            command.set_defaults(function=train_command)
        else:
            command.add_argument("--checkpoint", required=True)
            command.add_argument("--output", required=True)
            command.add_argument("--batch-size", type=int, default=256)
            command.set_defaults(function=infer_command)
    return parser


def main() -> None:
    args = build_parser().parse_args()
    args.function(args)


if __name__ == "__main__":
    main()
