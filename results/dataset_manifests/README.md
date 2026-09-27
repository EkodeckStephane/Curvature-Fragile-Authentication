# Dataset manifests

## Context

Real-image evaluation uses third-party datasets whose pixels remain under their
upstream terms. This directory stores reproducibility manifests that identify
the selected images without redistributing image pixels or local machine paths.

## Objective

The first restricted public pilot keeps only DTD and MS-COCO val2017 rows whose
source identities and reuse metadata are pinned.

The multicorpus provenance supplement records dataset-level source, version,
citation, reuse-status, split-count, and non-redistribution policy for the
five-subcorpus robustness aggregate. It does not publish image pixels,
watermarked images, attacked images, local paths, or secret key material.

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

`multicorpus_provenance_v1.csv` and `multicorpus_provenance_v1.json` cover the
five subcorpora used by `results/blind_v2_multicorpus_expanded_v1/`: BOSSBase
1.01, BOWS-2, DTD textures, INRIA Holidays, and MS-COCO val2017. These files
are dataset-level provenance records. They intentionally avoid per-image
filenames for the five-subcorpus aggregate because each source dataset retains
its original terms; exact image identities remain in the private local map.

## Conclusion

This manifest is suitable for a restricted public real-image pilot in which
readers obtain images from the official dataset sources and reproduce the
selection through the published identifiers and hashes.

The multicorpus provenance supplement is suitable for submission review of the
sanitized aggregate: it states the evidence and rights status for every
subcorpus while preserving the no-redistribution policy for third-party pixels
and derivatives.
