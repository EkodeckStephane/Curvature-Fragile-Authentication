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

The theory gate now distinguishes metric anisotropy from genuine Riemannian
curvature. The first implemented primitive is the generalized Fisher
sensitivity problem `F v = lambda P v`, where `P` is an explicit perturbation
cost metric.

## Results

The project is at the theory-feasibility and reproducibility-scaffold stage.
The initial audit established that a dominant eigenvector of a Fisher
information matrix maximizes a local quadratic sensitivity under a specified
norm, while Riemannian curvature is a separate object involving variation of
the metric. This distinction now constrains the method design and the wording of
the future theorem. A reproducible Semantic Scholar discovery run produced 459
unique literature candidates, including direct TIFS fragile-watermarking work
and an earlier application of Fisher information to blind watermark design.
A priority corpus of 22 full texts has now been read, including ten TIFS
articles and the foundational monograph *Methods of Information Geometry*.
The review identified a direct 2001 precedent that optimizes blind-watermark
synchronization patterns through Fisher information, and confirmed that the
Gaussian location--scale manifold has constant Fisher--Rao curvature. These
findings narrow the defensible contribution to a detector-linked geometric
design rule with matched empirical validation. A first source module now
implements the generalized Rayleigh quotient, Gaussian mean-family Fisher
matrix, generalized eigen-directions and spectral-gap calculation. A second
source module implements the frozen V2 blind-mapping primitives: luminance
conversion, block DCT, reserved-coefficient masking, local Fisher/cost
construction, HMAC authentication bits, dithered QIM, block scoring, threshold
selection and matched basis ablations. The V2 Fisher/smallest/identity-cost
modes use the analytic diagonal solution of `F v = lambda P v`, avoiding a
dense eigensolver for every block. Manuscript citations still require final
publisher-version and editorial-status checks.

Public empirical tables are limited to experiments whose commands, raw outputs,
configurations, seeds, environment details, and hashes are committed under
`results/`. Real-image dataset pilots are promoted only after the corresponding
source, version, license, split, and citation manifests are frozen.

## Conclusions

The Fisher metric supplies a defensible local sensitivity objective, subject to
an explicit perturbation-cost metric. A curvature-based authentication result
requires an additional geometric construction and evidence that it changes the
blind watermarking design. The current executable path is a detector-linked
Fisher-sensitivity prototype, validated first on synthetic gates and prepared
for a controlled real-image pilot after dataset provenance checks.

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
`results/sota/`. Re-running the PDF inventory preserves review metadata only
when both the relative path and SHA-256 are unchanged; a replaced file is reset
to the unverified state.

### Verify the current source tools

```powershell
python -m pip install -r requirements.txt
python -m unittest discover -s tests -v
```

### Reproduce the synthetic Fisher gate

```powershell
python scripts/run_synthetic_fisher_gate.py --config configs/synthetic_fisher_gate.json --output results/synthetic_fisher_gate/summary.json
```

The command validates the generalized Fisher-sensitivity primitive on known and
random symmetric-positive-definite cases, then writes an auditable summary under
`results/synthetic_fisher_gate/`.

### Implemented blind-mapping primitives

The current V2 prototype exposes the audited pieces needed before a real-image
pilot: orthonormal 16x16 DCT,
canonical reserved-coefficient masking, diagonal Fisher/cost matrices,
HMAC-SHA-256 authentication bits, dithered scalar QIM, block score extraction,
threshold calibration, pixel-map expansion, and Fisher/random/fixed matched
basis variants. Image-dataset pilots are prepared behind dataset licenses,
splits, attacks, and thresholds that are explicitly frozen before publication.

### Reproduce the synthetic image gate

```powershell
python scripts/run_blind_v2_synthetic_image_gate.py --config configs/blind_v2_synthetic_image_gate.json --output results/blind_v2_synthetic_image_gate/summary.json
```

The command generates deterministic synthetic luminance images, embeds and
verifies the V2 blind detector, selects `Delta_embed`, calibrates `tau`, runs
matched basis variants and block-aligned synthetic attacks, and writes a public
manifest without storing the secret key.

### Audit a local image corpus

```powershell
python scripts/audit_image_corpus.py --root path\to\image_corpus --output results\_scratch\image_corpus_audit.json
```

Dataset-derived summaries stay out of public results until source, version,
license, splits, and citation requirements are checked.

When a derived multicorpus was produced with deterministic source-dependent
filenames, reconstruct the local identity map with:

```powershell
python scripts/map_multicorpus_identities.py --derived-root path\to\multicorpus_v1_0 --output-csv results\_scratch\multicorpus_identity_map.csv --bossbase-root path\to\BOSSbase --bows2-root path\to\bows2\cover --dynacis-root path\to\external_processed_root
```

The command records hashes, dimensions, matched source identifiers, split
labels, and original paths when the upstream raw or processed roots are
available. The resulting CSV remains private unless every retained dataset has
publishable source, version, license, split, and citation metadata.

After the local identity map exists, enrich it with available dataset metadata:

```powershell
python scripts/enrich_multicorpus_metadata.py --identity-map results\_scratch\multicorpus_identity_map.csv --output-csv results\_scratch\multicorpus_identity_map_enriched.csv --coco-annotations path\to\annotations\instances_val2017.json
```

For COCO, the command attaches image IDs and license names/URLs from the
official annotation JSON. For DTD, it records the official source page, package
version and citation used by the provenance gate.

Finally, build the public restricted DTD+COCO pilot manifest:

```powershell
python scripts/build_public_pilot_manifest.py --enriched-map results\_scratch\multicorpus_identity_map_enriched.csv --output-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --output-json results\dataset_manifests\dtd_coco_public_pilot_v1.json
```

The public manifest contains source identifiers, hashes, citations, license or
reuse metadata, dimensions and split labels. It excludes machine-local paths,
dataset pixels, watermarked images and attacked images.

### Run a private real-image scratch pilot

```powershell
python scripts/run_blind_v2_real_pilot.py --root path\to\image_corpus --output results\_scratch\blind_v2_real_pilot.json --calibration-images-per-subcorpus 1 --evaluation-images-per-subcorpus 1
```

This command exercises the V2 detector on local images and writes an ignored
engineering manifest. Delta and threshold calibration use the calibration
split; clean and attack metrics use the evaluation split. The manifest is
marked `promotionReady: false` until the dataset manifest is complete.

For the restricted DTD+COCO pilot manifest, resolve only the declared images:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root path\to\multicorpus_v1_0 --output results\_scratch\blind_v2_manifest_pilot.json --calibration-images-per-subcorpus 1 --evaluation-images-per-subcorpus 1 --hash-files
```

When `--manifest-derived-root` is provided, the runner verifies the manifest
SHA-256 hashes for the local derived files. Alternatively, official dataset
roots can be supplied with `--dtd-root` and `--coco-root`.

Threshold calibration is global by default. For private stratified diagnostics,
`--threshold-scope subcorpus` calibrates one decision threshold per subcorpus
while keeping the same calibration/evaluation split discipline.

The detector score can be switched for private design diagnostics:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root path\to\multicorpus_v1_0 --output results\_scratch\blind_v2_manifest_top4.json --calibration-images-per-subcorpus 3 --evaluation-images-per-subcorpus 5 --target-false-positive-rate 0.01 --score-mode fisher_top4
```

Available score modes are `hamming`, `fisher_weighted`, `fisher_top4`, and
`fisher_reliability`. The non-default modes are experimental diagnostics for
coupling the Fisher sensitivity values to the block decision rule.
`fisher_reliability` estimates per-bit clean decoding error rates on the
calibration images, then scores mismatches with weights proportional to
`Fisher sensitivity / clean-error risk`. Threshold calibration uses observed
clean score values, so continuous weighted scores are not restricted to the
Hamming `k/8` grid.

The authenticated feature quantization can also be varied upstream:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root path\to\multicorpus_v1_0 --output results\_scratch\blind_v2_manifest_feature_step8.json --calibration-images-per-subcorpus 3 --evaluation-images-per-subcorpus 5 --target-false-positive-rate 0.01 --auth-feature-step 8.0
```

`auth-feature-step` controls the quantization granularity of canonical features
before HMAC authentication. Larger values are expected to improve clean
stability, with the empirical attack sensitivity checked by the same private
pilot summaries and paired comparisons.

The feature set itself can be switched from the full canonical vector to
Fisher-sensitive radial bands:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root path\to\multicorpus_v1_0 --output results\_scratch\blind_v2_manifest_sensitive_bands.json --calibration-images-per-subcorpus 3 --evaluation-images-per-subcorpus 5 --target-false-positive-rate 0.01 --auth-feature-step 16.0 --auth-feature-mode sensitive_bands --auth-feature-band-count 2
```

`auth-feature-mode=sensitive_bands` authenticates only the DCT radial bands
selected by the local Fisher sensitivity model. This is an upstream diagnostic
for testing whether Fisher-guided feature selection changes the tamper detector
relative to matched basis baselines.

The current predeclared candidate protocols are stored in:

```text
configs/blind_v2_sensitive_bands_candidate_v1.json
results/analysis_plans/blind_v2_sensitive_bands_candidate_v1.md
configs/blind_v2_sensitive_bands_candidate_v2.json
results/analysis_plans/blind_v2_sensitive_bands_candidate_v2.md
```

Both fix `auth-feature-step=24.0`, `auth-feature-mode=sensitive_bands` and
`auth-feature-band-count=3`. Candidate v2 additionally fixes
`delta-embed-candidates=8.0` for the next license-clean validation stage.

The embedding allocation can also be switched:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root path\to\multicorpus_v1_0 --output results\_scratch\blind_v2_manifest_delta_alloc.json --calibration-images-per-subcorpus 3 --evaluation-images-per-subcorpus 5 --target-false-positive-rate 0.01 --delta-mode fisher_sqrt
```

`delta-mode=constant` is the default. `delta-mode=fisher_sqrt` keeps the same
payload and mean base step scale while assigning larger QIM steps to directions
with larger model sensitivity values. It is an experimental design diagnostic.

```powershell
python scripts/summarize_real_pilot.py --input results\_scratch\blind_v2_real_pilot.json --output-dir results\_scratch\blind_v2_real_pilot_summary
```

The summarizer converts a private scratch manifest into ignored CSV and
Markdown tables for engineering review.

When the input manifest contains clean decoding diagnostics, the summarizer
also writes `clean_bits.csv`, with per-split, per-basis and per-bit mismatch
counts, totals and clean error rates. This supports detector-aware reliability
diagnostics while keeping pixels, local paths and secret material out of the
public repository.

For finer clean-stability diagnostics by subcorpus and reserved DCT bit:

```powershell
python scripts/analyze_clean_bit_stability.py --root path\to\image_corpus --output-json results\_scratch\clean_bit_stability.json --output-csv results\_scratch\clean_bit_stability.csv --calibration-images-per-subcorpus 1 --evaluation-images-per-subcorpus 1
```

The diagnostic writes private ignored JSON/CSV artifacts with
`basis × split × subcorpus × bit` clean mismatch rates. It stores no pixels,
local image paths, or secret key material.

For paired Fisher-vs-baseline diagnostics on the same evaluated images, run:

```powershell
python scripts/paired_compare_real_pilot.py --input results\_scratch\blind_v2_manifest_pilot.json --output-dir results\_scratch\blind_v2_manifest_pilot_paired
```

The paired comparison reports per-attack and clean-image deltas, bootstrap
intervals and sign-flip permutation p-values. These diagnostics remain private
until a public analysis plan and promoted result manifest are frozen.

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
