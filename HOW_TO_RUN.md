# How to run and use DeepOCR

DeepOCR turns handwritten and printed documents (images or PDFs) into structured, editable text.
It runs entirely on your own computer, in a web browser.

Team: Nalin Sharma (CSJMA23001390292), Amritansh Singh Gaur (CSJMA23001390309),
Vineet Kumar Singh (CSJMA23001390311), Rajyavardhan Radhey (CSJMA23001390322).

---

## 1. What you need

| Requirement | Details |
| --- | --- |
| Computer | Windows 10 or 11, 8 GB RAM or more (16 GB recommended). No GPU needed. |
| Python | **Python 3.12** (3.10 and 3.11 also work) from <https://www.python.org/downloads/>. During setup, tick **"Add python.exe to PATH"**. |
| Disk space | About 4 GB (Python packages and models). |
| Internet | Needed **only the first time**, to install Python packages and download the pretrained TrOCR and PaddleOCR models. After that it works offline. |
| Node.js | **Not needed.** The web interface is already built (`web/dist`). Only needed if you change the frontend code. |

---

## 2. Start the application (first time)

1. **Unzip** `DeepOCR_Project.zip` to a short path, for example `C:\DeepOCR`.
   Avoid deep folders such as `Desktop\New folder\...`: Windows has a 260-character path limit.
2. Open the `DeepOCR` folder and **double-click `start.bat`**.
   - The first run installs the Python packages from `web-requirements.txt` (PyTorch, OpenCV,
     PaddleOCR, Transformers, FastAPI and others). This takes **5–15 minutes**, depending on the internet speed.
   - If it does not install everything, open PowerShell in the folder and run:
     ```powershell
     python -m pip install -r web-requirements.txt
     ```
3. When you see `Starting DeepOCR at http://127.0.0.1:8000`, the browser opens automatically.
   If it does not, open **<http://127.0.0.1:8000>** yourself.
4. Keep the black console window open while you use the app. Press **Ctrl + C** in it to stop the server.

**Every later time:** just double-click `start.bat`. It starts in a few seconds.

> The **first document** you read with each engine is slower (10–40 seconds), because the model is loaded
> (and, on the very first run, downloaded). After that, models stay in memory.

### Manual start (alternative)

```powershell
cd C:\DeepOCR
python -m pip install -r web-requirements.txt
python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

Then open <http://127.0.0.1:8000>. To use another port: `.\start.ps1 -Port 8080`.

---

## 3. Using the web app

The top bar has four pages: **Studio**, **Compare**, **Research** and **How it works**.

### Studio: read a document

1. **Add a document:** drop a file on the box, click *browse*, paste a screenshot with **Ctrl + V**,
   or use **Camera** on a phone. PDF, PNG, JPG, TIFF, BMP and WEBP up to 100 MB / 100 pages.
   *Or click one of the four sample documents at the bottom; they run immediately.*
2. **Choose an engine.** Each one shows its measured error rate on handwriting and print, and its speed:
   - **Consensus · 5 models**: most accurate. All five models read every line and vote on each word (about 1 s per line).
   - **TrOCR · fine-tuned**: best balance for handwriting.
   - **CRNN / CNN-BiLSTM**: our models trained from scratch; very fast, good on print.
   - **TrOCR** (pretrained) and **PaddleOCR**: reference engines.
3. **Options:**
   - *Image cleanup*: **Auto** (straighten + contrast), **Scan** (also removes noise),
     **Photo** (also evens out shadows), **Off**.
   - *Auto-correct*: fixes typical OCR mistakes. Every change is listed and can be undone.
4. Click **Extract text** (or press **Ctrl + Enter**). The progress card shows each stage live;
   with Consensus you can see each model reading the lines. **Cancel** stops it.
5. **Review the result:**
   - The summary strip shows agreement (or confidence), lines, words, disputed words and the time.
   - The **document viewer** colours each line: *Layout* (headings, tables, checkboxes, signature fields),
     *Agreement* (green = models agree, amber = partly, red = disputed) or *Confidence*.
     *Before / after* compares the page with its cleaned version. Hover a region to read it; click it to jump to its line.
   - **Lines** tab: underlined words are words the models disagreed on. **Click one** to see what the other
     models read and their share of the vote, then click **Use this** to replace it. The pencil edits a whole line.
   - **Review** tab: everything that needs a human check (disputed words, unclear lines, tables,
     checkboxes, signature fields), each with one-click actions.
   - **Text** tab: formatted preview, or edit the Markdown directly. **JSON** tab: the full structured result.
   - The bar above the page lists the **automatic fixes**; click **Undo** to restore the original word.
6. **Export:** **Copy**, **Markdown (.md)**, **Plain text (.txt)** or **JSON** (with coordinates).
   Use **Re-run with …** to read the same document with another engine, and **New document** to start again.

### Compare: engines side by side

1. Pick **Handwritten note** or **Noisy skewed scan** (their correct text is included), or upload your own file.
2. Select the engines (chips) and click **Run**. Engines run one after another.
3. Each column shows **CER, WER and time**, marks the **Most accurate** and **Fastest**, and highlights misread
   words in red (hover to see the correct word) and missing words with ‸.
   For your own file, click **Ground truth → Add text** and paste the exact text to get scores.

### Research: all experimental results

Headline numbers, the model comparison (CER / WER / accuracy / speed), the voting results, training and
fine-tuning curves, robustness charts, sample predictions, the architectures and the datasets.

### How it works

The eight pipeline stages, with the code file that implements each one.

### On a phone

Open the same address from a phone on the same network only if you started the server with
`--host 0.0.0.0` (by default it accepts connections from this computer only). On narrow screens the app
switches between *Document* and *Text*, and Export moves to the bottom bar.

---

## 4. Suggested 5-minute demo

1. **Studio →** click the **Handwritten note** sample (Consensus engine). Show the live progress with five models.
2. On the result, switch the overlay to **Agreement**, then click an underlined word in **Lines**
   (for example *“(Karlophane),”*) and show the alternatives and their vote shares. Click **Use this**.
3. Open the **Review** tab, then **Export → Markdown**.
4. **New document →** click **Application form** (PaddleOCR). Switch the overlay to **Layout** and show the
   checkboxes, table and signature field, and the formatted **Text** preview.
5. **Compare →** click **Handwritten note**: all engines are scored word by word against the true text.
6. **Research →** show the comparison chart, the voting section and the robustness charts.

---

## 5. Project files in this package

| Folder / file | What it is |
| --- | --- |
| `start.bat`, `start.ps1` | One-click launcher |
| `api/` | FastAPI server: decoding, preprocessing, engines, consensus, layout, endpoints |
| `models/`, `training/`, `datasets/` | Model architectures, training code, data loading and augmentation |
| `postprocessing/` | Error correction and consensus voting (ROVER) |
| `checkpoints/` | **Trained models**: CNN-BiLSTM, CRNN and the fine-tuned TrOCR |
| `results/` | Evaluation results, training histories and plots |
| `samples/` | The four demo documents (with ground truth) |
| `web/` | React + TypeScript web interface (`web/dist` is the built app) |
| `app/streamlit_app.py` | Streamlit interface, as in the proposal |
| `scripts/` | Data preparation, training, fine-tuning, evaluation, robustness |
| `report/` | **Project report** (`DeepOCR_Project_Report.docx` and `.pdf`) |
| `presentation/` | **Presentation** (`DeepOCR_Presentation.pptx` with Morph transitions, and `.pdf`) |
| `README.md` | Technical overview, results and reproduction steps |

The training datasets (`data/`) are not included because of their size. They are only needed to retrain
or re-evaluate the models; see section 7.

---

## 6. Presentation and report

- **Presentation:** open `presentation\DeepOCR_Presentation.pptx` in PowerPoint and press **F5**.
  The Morph transitions need PowerPoint 2019, 2021 or Microsoft 365. Speaker notes are under *View → Notes*.
  `DeepOCR_Presentation.pdf` is a static copy.
- **Report:** `report\DeepOCR_Project_Report.docx` (editable) and `DeepOCR_Project_Report.pdf`.
  After editing in Word, right-click the table of contents and choose **Update Field → Update entire table**.

---

## 7. Optional: tests, Streamlit, and retraining

```powershell
# Automated tests (114 backend tests)
python -m pip install -r requirements.txt
python -m pytest -q

# Frontend tests and rebuild (needs Node.js 20+)
cd web
npm ci
npm test
npm run build

# Streamlit interface from the proposal
python -m streamlit run app/streamlit_app.py --server.address 127.0.0.1

# Recreate the datasets and retrain (several hours on a CPU)
python scripts/download_iam_hf.py
python scripts/prepare_dataset.py --dataset synthetic --n_samples 6500
python scripts/train.py --model cnn_bilstm --epochs 11 --max_minutes 170
python scripts/train.py --model crnn --epochs 9 --max_minutes 170
python scripts/finetune_trocr.py --steps 300
python scripts/evaluate.py --all --limit 600
python scripts/ensemble_eval.py
python scripts/robustness.py --lines 80
```

---

## 8. Troubleshooting

| Problem | Fix |
| --- | --- |
| `Python 3.10+ is required` | Install Python 3.12 and tick *Add python.exe to PATH*, then restart `start.bat`. |
| `start.bat` closes immediately | Open PowerShell in the folder and run `.\start.ps1` to see the message. If scripts are blocked: `powershell -ExecutionPolicy Bypass -File start.ps1`. |
| Package installation fails | Run `python -m pip install --upgrade pip`, then `python -m pip install -r web-requirements.txt`. |
| An engine shows *unavailable* | Its package is missing (PaddleOCR or Transformers): install `web-requirements.txt` again. TrOCR and PaddleOCR need internet the first time to download their models. |
| “Address already in use” | Another server is running. Close it, or start on another port: `.\start.ps1 -Port 8080`. |
| The first extraction is slow | Normal: the model is loading. Later documents are much faster. |
| “Another document is being processed” | The server reads one document at a time; the app waits and retries automatically. |
| Unzip errors or missing files | Unzip to a short path such as `C:\DeepOCR`. |
