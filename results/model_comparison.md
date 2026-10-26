| Model | Architecture | IAM handwriting CER | WER | Acc | Synthetic printed + script CER | WER | Acc | ms/line |
|---|---|---|---|---|---|---|---|---|
| CNN + BiLSTM (baseline) | CNN-BiLSTM-CTC | 21.79 | 61.97 | 1.00 | 4.85 | 17.49 | 66.67 | 12.2 |
| CNN + BiLSTM (baseline) + post-processing | CNN-BiLSTM-CTC | 20.61 | 51.37 | 4.83 | 4.60 | 15.37 | 69.83 | 42.5 |
| CRNN | CRNN (VGG + BiLSTM + CTC) | 21.66 | 61.47 | 0.50 | 5.25 | 18.54 | 63.17 | 15.9 |
| CRNN + post-processing | CRNN (VGG + BiLSTM + CTC) | 20.55 | 50.99 | 3.17 | 5.00 | 16.13 | 67.00 | 40.5 |
| TrOCR Transformer (zero-shot) | ViT encoder + Transformer decoder | 6.97 | 16.40 | 40.67 | 20.37 | 54.68 | 53.83 | 377.3 |
| TrOCR Transformer (fine-tuned) | ViT encoder + Transformer decoder | 6.73 | 16.79 | 37.83 | 2.59 | 10.81 | 84.33 | 415.2 |
| PaddleOCR | PP-OCR recogniser | 28.84 | 72.50 | 1.50 | 3.89 | 21.09 | 66.17 | 139.2 |
