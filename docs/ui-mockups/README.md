# DeepOCR UI/UX redesign (light mode)

Mockups in this folder are rendered from real project data: actual sample documents,
detected line polygons, model confidences, test-set results, and training and
robustness curves. `mockup.css` holds the design tokens the implementation will use.

| File | Screen |
| --- | --- |
| `01-studio-upload.jpg` | Studio: new extraction |
| `02-studio-processing.jpg` | Studio: processing |
| `03-studio-results-handwriting.jpg` | Results: handwriting, confidence overlay, Lines tab |
| `04-studio-results-form.jpg` | Results: form, layout overlay, Text tab, auto-fix and review bar |
| `05-compare.jpg` | Compare engines |
| `06-research.jpg` | Research dashboard |
| `07-mobile-results.jpg` | Results on a phone |

## Principles

1. **Light only.** Remove dark mode, the theme toggle, and the aurora, glow and glass effects.
2. **The document comes first.** The page sits on a neutral stage, and everything else
   supports reading and checking it.
3. **Colour means something.** Confidence uses green, amber and red; document structure has
   one tint per kind; each model has one identity colour (charts only); indigo marks actions.
4. **Real numbers, honestly labelled.** Error rates come from held-out tests; confidence is never presented as accuracy.
5. **Progressive disclosure.** Sensible defaults come first; detail is one click away.

## Design system

- **Colour:** canvas `#F5F6F8`, surfaces white, borders `#E4E7EC`/`#D0D5DD`, text `#101828` / `#475467` / `#667085`, accent indigo `#4F46E5` (soft `#EEF2FF`), and success, warning and danger pairs.
- **Type:** Space Grotesk for headings; Inter for UI text with tabular numerals for every metric; JetBrains Mono only for code and JSON.
- **Shape:** radius 6 / 10 / 14, four soft shadow levels, 4-pt spacing grid, 1440 desktop and 390 phone layouts.
- **Components:** button (primary, secondary, ghost, small), badge, segmented control, switch, tabs, card, confidence chip, meter, tooltip, toast, stepper.

## New: consensus vote ("Consensus · 5 models")

All five models read every line; their words are aligned and decided by a weighted vote
(ROVER), where each vote counts *model weight × that model's confidence*. Words several
models agree on win, and words with no majority are flagged for review. Measured with 2-fold
cross-validation on the held-out test lines, the vote is the most accurate option:

| | Handwriting CER / WER | Print CER / WER |
| --- | --- | --- |
| Best single model (TrOCR fine-tuned) | 6.73% / 16.79% | 2.59% / 10.81% |
| **Consensus vote** | **5.74% / 14.24%** | **2.35% / 5.86%** |

On the handwritten demo page it cuts character errors from 7.4% to 4.4%. It is the slowest
option (it runs every model, about 1 s per line on this CPU), so the fine-tuned TrOCR stays
available as the "best balance" choice.

UI impact (shown in the mockups):
- **Upload:** Consensus appears as the "Most accurate" engine.
- **Processing:** each model's progress is shown, then a "Vote on each word" step.
- **Results:**
  - An **Agreement** overlay and agreement score per line.
  - Disputed words are underlined; clicking one shows every alternative with its vote
    share and **Use this** to swap it in.
  - The Review tab counts disputed words.
- **Compare:** a Consensus column.
- **Research:** a "Model voting" section with a real example from the test page.

## Screen changes

**Studio: upload**
- A single "New extraction" card in three numbered steps: Document, Engine, Options.
- The engine picker is a list that shows each model's measured handwriting and print error
  rates, its speed per line, and a *Recommended* tag, replacing five large cards.
- Image cleanup becomes presets (Auto / Scan / Photo / Off) instead of four separate toggles.
  Auto-correct is a switch with a note that every change can be undone.
- The left column states the value with real metrics and a short "What happens to your
  document" flow. Samples sit in a row of document cards.

**Studio: processing**
- A focused card showing the document with detected lines appearing, and a vertical stepper
  with real per-stage times and "reading line *n* of *m*".
- An elapsed-time hint and a Cancel button.

**Studio: results**
- The header shows the file, *Re-run with ▾* and *Export ▾* (Markdown, text, JSON, copy).
- A compact strip shows confidence, lines, words, review count, auto-fixes and time, with a time breakdown.
- Three columns: a page rail with cleanup info; the document with Layout / Confidence / Off
  overlays, zoom, and the legend under the page; and a text panel.
- Text panel tabs:
  - **Text:** Preview or Markdown.
  - **Lines:** confidence per line, click to locate, inline edit.
  - **Review:** a queue with accept and edit actions.
  - **JSON.**
- Hovering or selecting a line highlights it on the page, and the reverse also works.
- An auto-fix bar lists each automatic correction with *Undo* and offers *Start review* for flagged items.

**Compare**
- One setup toolbar: the document, engine chips, ground-truth status, and *Run*.
- The diff becomes a **whole-document, word-level diff**. This fixes the current bug where
  engines that split lines differently misalign the comparison. Misread words are underlined
  with the correct word in a tooltip; missing words show a ‸ caret.
- *Most accurate* and *Fastest* badges.

**Research**
- A report layout with a sticky "On this page" navigation and KPI cards.
- Ranked bar charts, the results table, training and fine-tuning curves, and robustness charts
  (including rotation with and without deskew), all in one consistent palette.

**How it works:** the same seven pipeline stages, restyled with the new components.

**Mobile:** Document / Text / Lines tabs and a sticky Copy / Export bar.

## Backend support needed

- **Live progress:** `POST /api/v1/jobs` returns a job id, and `GET /api/v1/jobs/{id}` returns the
  stage, lines done / total and stage timings. The existing synchronous `/api/v1/ocr` stays.
- **Cleanup presets:** Auto applies deskew and contrast; Scan applies deskew, denoise and
  contrast; Photo applies deskew, denoise, contrast and binarise; Off applies nothing.
- **Undo for corrections:** handled in the client using the original text kept for each block. No API change.
- **Consensus engine (done):** `engine=ensemble` returns per-word votes (`words`: text, support,
  sources, alternatives) on every region and block. Implementation is in `postprocessing/ensemble.py`
  and `api/engines.py`; the weights come from `scripts/ensemble_eval.py`.

## Implementation phases

1. Tokens and base components (replaces `styles.css`, removes dark mode).
2. Studio upload and processing, plus the job-progress API.
3. Results workspace (viewer, linked lines, review queue, undo, export menu).
4. Compare (word diff), Research and How it works.
5. Responsive and accessibility pass (keyboard, focus, ARIA, reduced motion), updated tests,
   build, and verification in the browser.
