# DeepOCR — Faculty Demonstration Guide

A 10-minute live demo script with talking points and fallbacks.

## Before the presentation (do this once, with internet)

1. Open PowerShell in the project folder and run `.\start.ps1` (or double-click `start.bat`).
   The first start installs missing packages and builds the UI.
2. In the browser, run each demo once: **Studio → One-click demos → Handwritten note**,
   then choose **TrOCR** and press **Re-run**. This caches the PaddleOCR and TrOCR model
   weights, so the live demo works offline and starts fast.
3. Open the **Research** page and check that the comparison table, training curves and
   robustness charts are filled in.
4. Keep this guide and `results/model_comparison.md` open as a backup.

## Live demo flow

| # | Show | Say |
| --- | --- | --- |
| 1 | **Pipeline** page | "This is the architecture from our proposal: preprocessing, detection, layout analysis, deep-learning recognition and post-processing. Each stage maps to a module in the code." |
| 2 | **Studio → Handwritten note** demo, engine **DeepOCR · CRNN** | "This is real IAM handwriting from writers the model never saw in training. Our CRNN was trained from scratch on IAM plus synthetic data." Point out the live pipeline animation, then the per-stage timing bar. |
| 3 | Viewer: switch **Structure → Confidence** | "The heatmap shows the model's per-line confidence, so the reviewer knows where to look." Hover a line to show the tooltip. |
| 4 | Result tabs: **Lines** and **Review** | "Auto-correct fixes are shown as before → after, never silently. Low-confidence text becomes [Illegible] and joins the review queue." |
| 5 | **Re-run** with **TrOCR Transformer** | "The advanced model: a Vision Transformer encoder with a Transformer decoder, used through transfer learning. Notice the lower error on the same page." |
| 6 | **Studio → Application form** (PaddleOCR) | "Complex layout: heading detection, form fields, checkbox states rendered as ☑/☐, the ruled fee table rebuilt as a Markdown table, and the signature field flagged." Show the **Reader** tab. |
| 7 | **Studio → Noisy skewed scan** (deskew + denoise + contrast) | Use the **Before/after** slider and the **stage strip**: "The projection-profile deskew measured about 4° and straightened it before recognition." |
| 8 | **Compare** page: pick a demo, all engines, paste the true text | "The same page through four models, scored live with CER and WER, with character-level diffs." |
| 9 | **Research** page | Walk through the comparison table (CER, WER, accuracy, speed), training curves, the TrOCR fine-tuning curve, the robustness charts (CER as noise, blur and rotation increase) and the sample predictions. |
| 10 | Export | Edit a word in **Editor**, then **Export Markdown**: "Exports always use the reviewed text." |

## Key talking points

- **Data**: IAM lines with the official writer-independent split, plus 6,500 synthetic
  printed and handwriting-font lines with realistic degradations. Test sets are never
  used for training or model selection.
- **Why three models**: the CNN-BiLSTM baseline and the CRNN are both CTC models
  trained from scratch. TrOCR shows how much large-scale pretraining (transfer learning)
  helps with limited data and CPU-only compute.
- **Metrics**: CER and WER (lower is better), exact-line accuracy, and ms per line,
  measured on held-out test lines. Confidence is not accuracy.
- **Robustness**: CER is measured under increasing noise, blur, rotation, low contrast
  and JPEG compression, as promised in the proposal's "real-world robustness testing".
- **Engineering**: bounded uploads, automatic retry when busy, hot-reloading
  checkpoints, a fallback detector, and 100+ automated tests.

## If something goes wrong

| Symptom | Fix |
| --- | --- |
| Browser shows "API offline" | The server isn't running. Run `.\start.ps1` again. |
| An engine card is greyed out | No checkpoint (CTC models) or the model failed to load. Pick another engine; the error is in the card tooltip. |
| TrOCR is slow the first time | It loads about 250 MB of weights. Pre-run it before the demo (step 2 above). |
| "Another document is being processed" | One document is processed at a time. The UI retries automatically. |
| No internet | PaddleOCR and TrOCR work offline once their weights are cached (step 2). The CNN-BiLSTM and CRNN checkpoints are local. |
| PaddleOCR fails to import | The CTC engines fall back to the OpenCV MSER detector automatically. |

Stop the server with `Ctrl+C` in its PowerShell window.
