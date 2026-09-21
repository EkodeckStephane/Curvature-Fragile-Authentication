# Blind V2 sensitive-bands candidate protocol v2

## Context

The executable detector uses blind fragile watermarking with authentication
bits derived from local DCT-block features. Candidate v1 established the
current sensitive-band authentication rule. Private multicorpus diagnostics
then showed that a stronger fixed QIM embedding strength improves clean bit
stability and Fisher-vs-baseline attack localization on the engineering-only
five-subcorpus split.

## Objective

Evaluate one fixed candidate rule: authenticate Fisher-sensitive DCT radial
bands and use a fixed stronger embedding strength, then compare
Fisher-selected embedding directions against matched baselines at the same
payload, attacks, threshold rule and data split.

## Candidate protocol

Machine-readable config:

```text
configs/blind_v2_sensitive_bands_candidate_v2.json
```

Fixed candidate parameters:

```text
auth_feature_step = 24.0
auth_feature_mode = sensitive_bands
auth_feature_band_count = 3
score_mode = hamming
delta_mode = constant
delta_embed_candidates = [8.0]
target_false_positive_rate = 0.01
threshold_scope = global
```

Basis modes:

- `fisher`
- `smallest`
- `random`
- `fixed`
- `identity_cost`

Attacks:

- `center_mean`
- `copy_move`
- `constant_average_block`
- `inter_block_substitution`

## Dataset and split

Use the restricted public DTD+COCO manifest:

```text
results/dataset_manifests/dtd_coco_public_pilot_v1.csv
```

Declared split:

- 10 calibration images per subcorpus
- 10 evaluation images per subcorpus
- manifest row order determines calibration/evaluation membership within each
  subcorpus

Dataset pixels and local roots remain outside version control.

## Reproduction command template

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root <local-derived-root> --output results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --calibration-images-per-subcorpus 10 --evaluation-images-per-subcorpus 10 --hash-files --target-false-positive-rate 0.01 --auth-feature-step 24.0 --auth-feature-mode sensitive_bands --auth-feature-band-count 3 --delta-embed-candidates 8.0
```

Then summarize and compare:

```powershell
python scripts\summarize_real_pilot.py --input results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --output-dir results\_scratch\blind_v2_sensitive_bands_candidate_v2_summary
python scripts\paired_compare_real_pilot.py --input results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --output-dir results\_scratch\blind_v2_sensitive_bands_candidate_v2_paired
```

## Primary analysis

Primary metric:

- block-level F1 for each declared attack

Reference comparison:

- Fisher versus fixed, random and smallest basis modes

Supportive engineering pattern:

- Fisher mean attack F1 exceeds fixed, random and smallest on the declared
  attacks.
- Fisher clean PSNR remains comparable to fixed, random and smallest.
- Identity-cost is interpreted with its measured clean-PSNR cost.

Uncertainty reporting:

- paired bootstrap confidence intervals
- paired sign-flip permutation p-values
- favorable, unfavorable and tied image-pair counts

## Conclusion use

This protocol tests the fixed stronger-embedding candidate on the current
license-clean restricted DTD+COCO manifest. Manuscript-level quantitative
claims require promoted public result manifests and a dataset-provenance gate
for every evaluated corpus.
