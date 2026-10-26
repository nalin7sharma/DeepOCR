# DeepOCR Data Directory

Datasets are generated locally and are not committed to git. Recreate them with the
commands below; every split file uses the simple format
`relative/image/path.png<TAB>transcription`.

## Structure

```
data/
└── processed/
    ├── iam/                 # IAM handwriting lines (real handwriting)
    │   ├── images/{train,val,test}/*.png
    │   ├── train.txt  val.txt  test.txt
    │   └── charset.txt
    └── synthetic/           # Printed + handwriting-font lines with degradations
        ├── images/{train,val,test}/*.png
        ├── train.txt  val.txt  test.txt
        └── meta.tsv         # font and style (printed / script) per image
```

## IAM Handwriting Database (lines)

```bash
python scripts/download_iam_hf.py
```

This downloads the [Teklia/IAM-line](https://huggingface.co/datasets/Teklia/IAM-line)
export (MIT licence) from the Hugging Face Hub, with the official writer-independent
split: 6,482 train / 976 validation / 2,915 test lines. Writers in the test set never
appear in training. A manual download from the
[IAM website](https://fki.tic.heia-fr.ch/databases/iam-handwriting-database) (`lines.tgz`
and `ascii/lines.txt` extracted into `data/raw/IAM/`) is still supported. Check it with
`python scripts/prepare_dataset.py --dataset iam`.

## Synthetic printed + script dataset

```bash
python scripts/prepare_dataset.py --dataset synthetic --n_samples 6500
```

This renders 6,500 lines (5,005 train / 520 validation / 975 test) with 19 printed and
13 handwriting-style system fonts. The text comes from IAM *training* transcriptions
(never validation or test) plus document templates such as dates, amounts, phone
numbers, e-mail addresses and form labels. Each line is then degraded with uneven
lighting, ink bleed or fading, skew, blur, low contrast, Gaussian and salt-and-pepper
noise, JPEG compression, or inversion (dark mode).

## Using the data

`configs/dataset.yaml -> dataset.train_sets` lists the sources that `scripts/train.py`
mixes (default: `iam` and `synthetic`). Each source's `train.txt` is used for training,
`val.txt` for model selection, and `test.txt` only by `scripts/evaluate.py`.

## Custom dataset

Create a folder with `train.txt`, `val.txt` and `test.txt` in the format above, then
train on it with `python scripts/train.py --model crnn --datasets path/to/folder`.
Only the characters in `dataset.charset` are learned.
