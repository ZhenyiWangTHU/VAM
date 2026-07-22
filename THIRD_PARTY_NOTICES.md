# Third-party software and models

VAM uses upstream packages as dependencies and does not vendor their source code.

## scGPT

- Project: <https://github.com/bowang-lab/scGPT>
- License: MIT
- Citation: Cui et al., *scGPT: Towards Building a Foundation Model for
  Single-Cell Multi-omics Using Generative AI*.

The VAM adapter uses scGPT to derive sample-level embeddings from protein
abundance matrices. Users must obtain compatible scGPT model files separately.

## ESM-2

- Project: <https://github.com/facebookresearch/esm>
- License: MIT
- Citation: Lin et al., *Evolutionary-scale prediction of atomic-level protein
  structure with a language model*, Science (2023).

The ESM-2 weights are downloaded by the upstream `fair-esm` package.

UniProt records and cohort data are not distributed in this repository and
remain subject to their respective data-use terms.
