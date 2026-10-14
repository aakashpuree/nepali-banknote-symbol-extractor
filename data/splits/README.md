# Dataset splits

No train/validation/test split is committed because images lack physical-note
and capture-session metadata. Splitting adjacent captures at random would leak
near-identical notes into evaluation.
