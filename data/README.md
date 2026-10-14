# Dataset layout

`raw/images/` contains the 55 JPEG files supplied with this project. They have
not been resized, renamed, or altered. The original `dataset/` folder was
reorganised here so application code and future model data are separate.

| Location | Purpose | Commit policy |
| --- | --- | --- |
| `raw/images/` | Immutable source captures | Keep originals; never overwrite. |
| `annotations/` | Human-reviewed OCR/date or detector labels | Add only after verifying the printed field. |
| `splits/` | Train/validation/test manifests | Create only after note/capture-session identifiers are known. |

There are no bounding-box or text annotations. Do not train a detector until
labels identify the target region and splits prevent the same physical note or
session appearing in more than one partition.

Run `python scripts/inspect_dataset.py` after environment setup to create an
ignored inventory at `artifacts/inventory/dataset_inventory.json`.
