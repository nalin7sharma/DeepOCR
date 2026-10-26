# Project presentation

`DeepOCR_Presentation.pptx` is the 15-slide project presentation; `DeepOCR_Presentation.pdf` is a PDF copy.

- **Morph transitions** on every slide after the title. Shapes that move between slides share a name
  starting with `!!` (for example the indigo circle `!!lens`, the model cards `!!card-cnn` …, the
  statistics `!!stat-hw`, the screenshots `!!shot`), so PowerPoint glides them from one slide to the next.
  Play it in Slide Show mode in PowerPoint 2019, 2021 or Microsoft 365 to see Morph.
- **Speaker notes** on every slide (View → Notes).
- Numbers and charts are read from `results/`, so they match the report and the evaluation files.

Rebuild after results change:

```powershell
cd presentation/src
npm install
node build_deck.js                                             # writes the .pptx
powershell -ExecutionPolicy Bypass -File finalize_deck.ps1     # adds Morph, exports the PDF (needs PowerPoint)
```

`finalize_deck.ps1 -ImageDir <folder>` also exports one PNG per slide.
