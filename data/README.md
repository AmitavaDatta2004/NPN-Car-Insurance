# Data Directory

This directory holds dataset manifests, lightweight samples, and metadata.
Raw datasets, images, and processed data are stored **outside Git**.

## What is committed here

| Path | Contents |
|---|---|
| `manifests/` | Train/validation/test split manifests (CSV/JSON, no raw images) |
| `samples/` | Small authorised sample images for notebooks and tests (if licence permits) |
| `README.md` | This file |

## What is NOT committed

- `data/raw/` — raw downloaded dataset archives and images
- `data/interim/` — intermediate processed files
- `data/processed/` — final preprocessed datasets
- `data/local/` — local SQLite database and application uploads

All of the above are excluded by `.gitignore`.

## Dataset instructions

See `README.md` §11 (Dataset plan) for the full dataset list, sources, and download procedure.

Run `notebooks/00_environment_and_data_download.ipynb` on Google Colab to download
and organise datasets. Follow the instructions in that notebook to place data in the
correct local directories without exposing credentials.

## Checksums

Record SHA-256 checksums of downloaded archives in `docs/EXPERIMENT_LOG.md`
under the relevant experiment entry.
