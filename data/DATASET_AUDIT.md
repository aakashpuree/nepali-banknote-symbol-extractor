# Initial dataset audit

Audit performed on the supplied files before any image-content transformation.

| Check | Finding |
| --- | --- |
| Image count | 55 JPEG files |
| Camera-size images | 22 at 3120×4160 (12.38 MP) |
| Gallery-size images | 33 at 720×1600 (1.15 MP) |
| Exact duplicates | None, based on SHA-256 hashes |
| Annotations | None found |
| Visible notes sampled | NPR 5 and NPR 10 examples |
| Confirmed date field | Not established |

The two resolution groups and sequential capture names suggest that many images
may be related captures. Treat all captures of the same physical note or session
as one group when creating a later evaluation split. `scripts/inspect_dataset.py`
generates a fresh full manifest and validates that each image can be opened after
the local environment is installed.
