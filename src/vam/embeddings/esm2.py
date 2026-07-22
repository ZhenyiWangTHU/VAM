"""ESM-2 protein embedding extraction."""

from __future__ import annotations

import argparse
from collections.abc import Iterable

import pandas as pd
import torch


def _batches(
    records: list[tuple[str, str]], token_budget: int, max_length: int
) -> Iterable[list[tuple[str, str]]]:
    batch: list[tuple[str, str]] = []
    tokens = 0
    for name, sequence in records:
        sequence = sequence.replace(" ", "").upper()[:max_length]
        if batch and tokens + len(sequence) > token_budget:
            yield batch
            batch, tokens = [], 0
        batch.append((name, sequence))
        tokens += len(sequence)
    if batch:
        yield batch


def extract_esm2_embeddings(
    table: pd.DataFrame,
    name_column: str = "protein_name",
    sequence_column: str = "sequence",
    device: str | None = None,
    token_budget: int = 4096,
    max_length: int = 1022,
) -> pd.DataFrame:
    """Return mean-pooled layer-33 ESM-2 embeddings (dimensions x proteins)."""

    try:
        import esm
    except ImportError as exc:
        raise ImportError("Install the optional dependency with `pip install fair-esm`.") from exc

    missing = {name_column, sequence_column}.difference(table.columns)
    if missing:
        raise ValueError(f"Missing input columns: {sorted(missing)}")
    names = table[name_column].astype(str).str.strip()
    if names.duplicated().any():
        raise ValueError("Protein names must be unique")
    sequences = table[sequence_column].fillna("").astype(str).str.replace(" ", "", regex=False)
    empty = sequences.str.len() == 0
    if empty.any():
        raise ValueError(f"Empty protein sequences: {names[empty].tolist()[:5]}")
    records = list(zip(names, sequences))
    model, alphabet = esm.pretrained.esm2_t33_650M_UR50D()
    converter = alphabet.get_batch_converter()
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    model.eval().to(target)
    outputs: dict[str, object] = {}
    with torch.no_grad():
        for batch in _batches(records, token_budget, max_length):
            labels, _, tokens = converter(batch)
            tokens = tokens.to(target)
            lengths = (tokens != alphabet.padding_idx).sum(1)
            representations = model(tokens, repr_layers=[33], return_contacts=False)[
                "representations"
            ][33]
            for index, label in enumerate(labels):
                outputs[label] = representations[index, 1 : lengths[index] - 1].mean(0).cpu().numpy()
    result = pd.DataFrame(outputs)
    if not torch.isfinite(torch.as_tensor(result.to_numpy())).all():
        raise ValueError("ESM-2 produced non-finite embeddings")
    result.index.name = "embedding_dim"
    return result


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VAM-compatible ESM-2 embeddings.")
    parser.add_argument("--input", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--name-column", default="protein_name")
    parser.add_argument("--sequence-column", default="sequence")
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    table = pd.read_csv(args.input, sep="\t")
    extract_esm2_embeddings(
        table, args.name_column, args.sequence_column, args.device
    ).to_csv(args.output, sep="\t")


if __name__ == "__main__":
    main()
