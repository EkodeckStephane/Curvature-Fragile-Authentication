# Blind V2 sensitive-bands candidate v2 result

## Context

This result evaluates the blind V2 fragile-authentication detector on the
restricted public DTD+COCO pilot manifest. The detector embeds HMAC-derived
authentication bits in reserved DCT coefficients and verifies block-level
tamper maps without access to the original image.

## Objective

Test the predeclared candidate v2 protocol: Fisher-sensitive authentication
bands with `auth_feature_step=24`, `auth_feature_band_count=3`, and fixed
`Delta_embed=8`, compared against matched `smallest`, `random`, `fixed`, and
`identity_cost` basis baselines.

## Methods

- Protocol: `configs/blind_v2_sensitive_bands_candidate_v2.json`
- Analysis plan: `results/analysis_plans/blind_v2_sensitive_bands_candidate_v2.md`
- Dataset manifest: `results/dataset_manifests/dtd_coco_public_pilot_v1.csv`
- Split: 10 calibration and 10 evaluation images per subcorpus
- Attacks: `center_mean`, `copy_move`, `constant_average_block`,
  `inter_block_substitution`
- Primary metric: mean block-level F1 per attack

Reproduction template:

```powershell
python scripts/run_blind_v2_real_pilot.py --manifest-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --manifest-derived-root <local-derived-root> --output results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --calibration-images-per-subcorpus 10 --evaluation-images-per-subcorpus 10 --hash-files --target-false-positive-rate 0.01 --auth-feature-step 24.0 --auth-feature-mode sensitive_bands --auth-feature-band-count 3 --delta-embed-candidates 8.0
python scripts\summarize_real_pilot.py --input results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --output-dir results\_scratch\blind_v2_sensitive_bands_candidate_v2_summary
python scripts\paired_compare_real_pilot.py --input results\_scratch\blind_v2_sensitive_bands_candidate_v2.json --output-dir results\_scratch\blind_v2_sensitive_bands_candidate_v2_paired
```

## Results

Fisher has the highest mean block-level F1 among the five declared basis modes
on all four attacks:

| Attack | Fisher F1 | Best non-Fisher F1 |
|---|---:|---:|
| center_mean | 0.973 | 0.967 |
| copy_move | 0.973 | 0.967 |
| constant_average_block | 0.966 | 0.957 |
| inter_block_substitution | 0.953 | 0.940 |

The strongest paired evidence is versus `smallest`. Fisher-vs-random and
Fisher-vs-fixed deltas are positive but tie-heavy on 20 evaluation images.
Fisher clean PSNR is comparable to fixed, random, and smallest, and about
4.94 dB above `identity_cost`.

## Conclusion

Candidate v2 is the current restricted license-clean pilot result to carry
forward. The aggregate result is stored in `summary.json`; it contains no
pixels, local machine paths, image derivatives, or secret key material.

## License and data

Repository source code is BSD 3-Clause licensed. Dataset pixels remain outside
version control and retain their original dataset terms. The public manifest
records dataset citations, source identifiers, and reuse metadata used for this
restricted pilot.
