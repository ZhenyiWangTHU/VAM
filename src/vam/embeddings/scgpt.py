"""Thin scGPT adapter for sample-level protein-expression embeddings."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch


def extract_scgpt_embeddings(
    adata_path: str,
    checkpoint_dir: str,
    model_filename: str = "best_model.pt",
    batch_size: int = 8,
    device: str | None = None,
) -> tuple[np.ndarray, list[str]]:
    """Embed AnnData rows with scGPT while preserving their sample order."""

    try:
        import scanpy as sc
        import scgpt
        from scgpt.tokenizer import tokenize_and_pad_batch
        from scgpt.tokenizer.gene_tokenizer import GeneVocab
        from scgpt.trainer import prepare_data, prepare_dataloader
    except ImportError as exc:
        raise ImportError(
            "scGPT extraction requires the optional `scgpt` and `scanpy` packages."
        ) from exc

    checkpoint = Path(checkpoint_dir)
    target = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
    adata = sc.read(adata_path)
    sample_ids = adata.obs_names.astype(str).tolist()
    vocab = GeneVocab.from_file(checkpoint / "vocab.json")
    for token in ("<pad>", "<cls>", "<eoc>"):
        if token not in vocab:
            vocab.append_token(token)
    in_vocab = np.array([name in vocab for name in adata.var_names])
    if not in_vocab.any():
        raise ValueError("None of the input protein names occur in the scGPT vocabulary")
    adata = adata[:, in_vocab]
    genes = adata.var_names.astype(str).tolist()
    tokenized = tokenize_and_pad_batch(
        adata.X,
        np.asarray(vocab(genes), dtype=int),
        max_len=3001,
        vocab=vocab,
        pad_token="<pad>",
        pad_value=-2,
        append_cls=True,
        include_zero_gene=True,
    )
    config = SimpleNamespace(
        pad_token="<pad>",
        pad_value=-2,
        mask_value=-1,
        mask_ratio=0.0,
        explicit_zero_prob=False,
        include_zero_gene=True,
        use_batch_labels=False,
        DSBN=False,
        CLS=True,
        GEPC=False,
        ESC=False,
        amp=target.type == "cuda",
    )
    batch_ids = np.zeros(tokenized["values"].shape[0], dtype=int)
    prepared, _ = prepare_data(tokenized, tokenized, batch_ids, batch_ids, config=config, epoch=0)
    loader = prepare_dataloader(prepared, batch_size=batch_size, shuffle=False)

    model_args = json.loads((checkpoint / "args.json").read_text(encoding="utf-8"))
    model = scgpt.model.TransformerModel(
        len(vocab),
        model_args["embsize"],
        model_args["nheads"],
        model_args["d_hid"],
        model_args["nlayers"],
        nlayers_cls=model_args.get("n_layers_cls", 3),
        vocab=vocab,
        pad_value=-2,
        do_mvc=model_args.get("do_mvc", True),
        dropout=model_args.get("dropout", 0.2),
        pre_norm=model_args.get("pre_norm", False),
        fast_transformer=model_args.get("fast_transformer", False),
    )
    state = torch.load(checkpoint / model_filename, map_location=target, weights_only=True)
    model.load_state_dict(state)
    model.eval().to(target)
    embeddings: list[np.ndarray] = []
    with torch.no_grad():
        for batch in loader:
            gene_ids = batch["gene_ids"].to(target)
            values = batch["values"].to(target)
            output = model(
                gene_ids,
                values,
                src_key_padding_mask=gene_ids.eq(vocab["<pad>"]),
                CLS=True,
                MVC=False,
                ECS=False,
            )
            embeddings.append(output["cell_emb"].cpu().numpy())
    return np.concatenate(embeddings), sample_ids


def main() -> None:
    parser = argparse.ArgumentParser(description="Extract VAM-compatible scGPT embeddings.")
    parser.add_argument("--adata", required=True)
    parser.add_argument("--checkpoint-dir", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--model-filename", default="best_model.pt")
    parser.add_argument("--batch-size", type=int, default=8)
    parser.add_argument("--device", default=None)
    args = parser.parse_args()
    embeddings, sample_ids = extract_scgpt_embeddings(
        args.adata,
        args.checkpoint_dir,
        args.model_filename,
        args.batch_size,
        args.device,
    )
    np.save(args.output, embeddings)
    Path(args.manifest).write_text("\n".join(sample_ids) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
