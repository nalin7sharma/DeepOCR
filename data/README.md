# DeepOCR Data Directory

## Structure

```
data/
├── raw/
│   ├── IAM/               # IAM Handwriting Database (download separately)
│   │   ├── lines/         # Line-level segmented images
│   │   │   ├── a01/
│   │   │   └── ...
│   │   └── ascii/
│   │       └── lines.txt  # Ground-truth transcriptions
│   └── synthetic/         # Auto-generated synthetic dataset
│       ├── images/
│       └── labels.txt
└── processed/             # Preprocessed outputs (auto-generated)
```

## Option A: IAM Handwriting Database

1. Register (free) at: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database
2. Download `lines.tgz` and `ascii.tgz`
3. Extract both into `data/raw/IAM/`
4. Run: `python scripts/prepare_dataset.py --dataset iam`

## Option B: Synthetic Dataset (quick start, no registration)

```bash
python scripts/prepare_dataset.py --dataset synthetic --n_samples 2000
```

This generates PIL-rendered text images suitable for quick development.

## Dataset Configuration

Edit `configs/dataset.yaml` to switch between datasets or adjust paths.

## Custom Dataset Format

For a custom dataset, use the **simple format**:
- Images directory: `data/raw/custom/images/*.png`
- Labels file: `data/raw/custom/labels.txt`
  - Each line: `relative/image/path.png\tground_truth_text`

Then update `configs/dataset.yaml`:
```yaml
dataset:
  name: custom

synthetic:
  data_dir: data/raw/custom
```
