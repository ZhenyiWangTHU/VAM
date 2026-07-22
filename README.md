# VAM

Source code for **VAM: A Multimodal dynamical foundation model for
characterizing human aging dynamics and enabling virtual aging perturbation**.

This release provides protein and metabolite preprocessing adapters, the
canonical VAM training and inference pipeline, and Dynamic Network Biomarker
(DNB) score calculation. It intentionally excludes cohort data, model weights,
participant identifiers, Grad/SHAP interpretation, and virtual perturbation
analysis.

The manuscript preprint is available on
[Research Square](https://www.researchsquare.com/article/rs-9402213/v1).

## Method

VAM combines three sample-level inputs:

1. protein structure features: protein abundance multiplied by mean-pooled
   ESM-2 representations;
2. protein expression features: sample embeddings extracted with scGPT;
3. metabolite abundances, normalized consistently by the training and inference
   commands.

Each modality receives a learned sigmoid gate. The gated features are
concatenated and projected to a 256-dimensional VAM embedding. Training
optimizes chronological-age MAE together with reconstruction losses for all
three modalities. In this implementation, “attention” refers to these learned
modality gates, not cross-sample or Transformer cross-attention.

## Installation

```bash
python -m pip install -e .
```

Install optional foundation-model dependencies when generating inputs:

```bash
python -m pip install -e '.[embeddings]'
```

## Input contract

Training and inference use three CSV/TSV files with `sample_id` as the first
column. Rows are aligned explicitly by sample ID.

- sequence features: samples × ESM-2 feature dimensions
- expression features: samples × scGPT embedding dimensions
- metabolite features: samples × raw non-negative metabolite abundances

Feature order and dimensions are stored in the checkpoint. Never rely on the
row order of a standalone NumPy array; scGPT extraction writes a companion
sample manifest.

## Embedding extraction

ESM-2 input is a TSV containing unique `protein_name` and `sequence` columns:

```bash
vam-esm2 --input proteins.tsv --output esm2.tsv
```

scGPT requires an AnnData file and an upstream checkpoint directory containing
`args.json`, `vocab.json`, and `best_model.pt`:

```bash
vam-prepare-scgpt --input protein_abundance.tsv --output protein.h5ad \
  --statistics protein_scaler.json
vam-scgpt --adata protein.h5ad --checkpoint-dir scgpt_model \
  --output scgpt.npy --manifest sample_ids.txt
```

For another cohort, pass the training scaler with `--reuse-statistics` to
preserve protein order and normalization.

The upstream model implementations and weights are not copied into this
repository. See [THIRD_PARTY_NOTICES.md](THIRD_PARTY_NOTICES.md).

## Training and inference

Generate the synthetic example:

```bash
python examples/synthetic/generate.py
```

Train:

```bash
vam train \
  --sequence examples/synthetic/sequence.tsv \
  --expression examples/synthetic/expression.tsv \
  --metabolite examples/synthetic/metabolite.tsv \
  --labels examples/synthetic/labels.tsv \
  --validation-ids examples/synthetic/validation_ids.txt \
  --config configs/train.example.yaml \
  --checkpoint outputs/vam.pt
```

Infer:

```bash
vam infer \
  --sequence sequence.tsv --expression expression.tsv \
  --metabolite metabolite.tsv --checkpoint outputs/vam.pt \
  --output outputs/predictions.npz
```

The output contains `sample_ids`, `predicted_age`, and 256-dimensional
`embeddings`.

## DNB scores

DNB accepts compressed NumPy files. A reference input contains `embeddings`
(samples × features) and `feature_names`. A cohort input contains
`embeddings`, anonymous `sample_ids`, and optionally `stages`.

```bash
vam-dnb build-reference --input reference.npz --output reference_network.npz
vam-dnb score --reference reference_network.npz --input cohort.npz \
  --output dnb_scores.csv
vam-dnb aggregate --input dnb_scores.csv --output idnb.csv
```

Defaults reproduce the original analysis: reference-network p-value 0.01,
one-sided SSN tail p-value 0.25, minimum hub degree 3, top-five iDNB, and
maximum stage aggregation. Use `--two-sided` or `--stage-method` for alternative
analyses.

## Data and checkpoints

UK Biobank and other cohort data cannot be redistributed. The embeddings and DNB scores for illustration could be accessed through [Google Drive](https://drive.google.com/drive/folders/1ITULHAj_ksOpmupVopxRd5XYVFEsGkbC?usp=sharing).

## Acknowledgements

We sincerely thank the open-source **scGPT** and **ESM-2** projects, which were
key to constructing VAM.
