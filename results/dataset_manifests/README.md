# Dataset manifests

## Context

Real-image evaluation uses third-party datasets whose pixels remain under their
upstream terms. This directory stores reproducibility manifests that identify
the selected images without redistributing image pixels or local machine paths.

## Objective

The first restricted public pilot keeps only DTD and MS-COCO val2017 rows whose
source identities and reuse metadata are pinned.

## Methods

`dtd_coco_public_pilot_v1.csv` is generated from a private enriched identity
map using:

```powershell
python scripts/build_public_pilot_manifest.py --enriched-map results\_scratch\multicorpus_identity_map_enriched.csv --output-csv results\dataset_manifests\dtd_coco_public_pilot_v1.csv --output-json results\dataset_manifests\dtd_coco_public_pilot_v1.json
```

The CSV records source identifiers, split labels, dimensions, hashes, dataset
source URLs, citations, and license or reuse metadata. The JSON file summarizes
counts and license distribution.

## Results

Version `v1` contains 40 records: 20 DTD images and 20 MS-COCO val2017 images.
It contains no dataset pixels, attacked images, watermarked images, or local
machine paths.

## Conclusion

This manifest is suitable for a restricted public real-image pilot in which
readers obtain images from the official dataset sources and reproduce the
selection through the published identifiers and hashes.
