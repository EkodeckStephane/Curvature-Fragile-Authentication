# Curvature-Fragile-Authentication

Reproducible research code and results for blind fragile image watermarking
guided by Fisher--Rao information geometry.

## Context

Fragile watermarking supports image authentication and spatial localisation of
tampering by embedding authentication information that reacts to image
modification. Existing schemes commonly select blocks, bit planes, or transform
coefficients through fixed rules or content heuristics. This project studies
whether a statistical image model can provide a principled sensitivity geometry
for that design choice.

## Objective

The primary research question is whether Fisher-geometric selection improves
tamper detection and localisation at matched payload and matched perceptual
quality, compared with relevant fragile-watermarking baselines. The evaluation
targets natural images and scanned-document images under a closed,
version-controlled tampering protocol.

## Methods

The planned study has four linked components:

1. fit a declared local statistical model to transformed image patches;
2. derive a Fisher--Rao metric and a mathematically defined local sensitivity
   rule;
3. use that rule in a blind embedding and verification algorithm with an
   explicit adversary model and decision threshold;
4. compare paired outputs against implemented baselines at matched payload and
   quality, reporting detection, localisation, uncertainty, failure cases, and
   multiplicity-corrected inference.

The theory gate currently distinguishes metric anisotropy from genuine
Riemannian curvature. Terminology and the central formal result will be frozen
only after the curvature construction has an operational role in the detector.

## Results

The project is at the theory-feasibility and reproducibility-scaffold stage.
The initial audit established that a dominant eigenvector of a Fisher
information matrix maximizes a local quadratic sensitivity under a specified
norm, while Riemannian curvature is a separate object involving variation of
the metric. This distinction now constrains the method design and the wording of
the future theorem. A reproducible Semantic Scholar discovery run produced 459
unique literature candidates, including direct TIFS fragile-watermarking work
and an earlier application of Fisher information to blind watermark design.
These records are discovery evidence; manuscript citations require subsequent
publisher/DOI verification and full-text reading.

Empirical performance tables will appear here only after the corresponding
commands, raw outputs, configurations, seeds, environment manifest, and hashes
are committed under `results/`.

## Conclusions

The Fisher metric supplies a defensible local sensitivity objective, subject to
an explicit perturbation-cost metric. A curvature-based authentication claim
requires an additional geometric construction and evidence that it changes the
blind watermarking design. The next milestone is to close that theory gate and
run a preregistered synthetic validation.

## Repository layout

```text
configs/        Versioned experiment configurations
results/        Immutable run outputs, manifests, summaries, tables, figures
scripts/        Dataset retrieval, experiment, analysis, and regeneration entry points
src/            Audited implementation
tests/          Unit, property, integration, and reproducibility tests
```

Downloaded datasets, manuscript drafts, governance documents, and local session
records remain outside version control.

## Reproduction workflow

The executable environment and experiment entry points are being frozen during
the theory gate. The completed workflow will follow this sequence:

1. create the documented Python environment from the locked dependency file;
2. run the dataset verification script, which checks source, version, license,
   and archive hash before extraction;
3. run the deterministic preprocessing command with the published
   configuration and split manifest;
4. execute the embedding, attack, verification, and baseline commands with the
   published seeds;
5. regenerate statistical summaries, tables, and figures from raw outputs;
6. verify the result manifest and compare all file hashes.

Exact commands will replace this staged description as each executable step is
implemented and validated. This keeps the README aligned with operations that
have actually been run.

### Reproduce the literature-discovery stage

Set the Semantic Scholar key only in the current process, then run:

```powershell
$env:SEMANTIC_SCHOLAR_API_KEY = "your-key"
python scripts/s2_literature_search.py --queries configs/literature_queries.json --output literature/api --limit 100 --interval 1.10
python scripts/s2_build_shortlist.py --raw literature/api/raw --output literature/manifests/s2_candidates.csv --manifest literature/manifests/s2_candidate_build.json
python scripts/index_literature.py --input literature/inbox --output literature/manifests/pdf_inventory.csv
Remove-Item Env:SEMANTIC_SCHOLAR_API_KEY
```

The raw API records, downloaded papers, and reading notes remain local. The
non-sensitive run summary and response hashes are published under
`results/sota/`.

### Verify the current source tools

```powershell
python -m unittest discover -s tests -v
```

## Editorial target

The first target is IEEE Transactions on Information Forensics and Security.
The study is positioned as image authentication and information forensics. The
public artifact is designed to regenerate every quantitative table and figure
used in the manuscript.

## License

Source code in this repository is licensed under the BSD 3-Clause License; see
[`LICENSE`](LICENSE). Each external dataset and baseline retains its own
license. Dataset retrieval is enabled only after its provenance and license are
recorded in the relevant manifest.
