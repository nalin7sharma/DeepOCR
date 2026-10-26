# Project report

`DeepOCR_Project_Report.docx` is the project report; `DeepOCR_Project_Report.pdf` is the same report exported by Word.

Everything in it is generated from the project, so it can be rebuilt after results change or when the
cover page and front matter are added in the required format.

```powershell
python report/src/make_figures.py                 # diagrams and charts -> report/figures/
python report/src/capture_screens.py              # app screenshots (server must be running on port 8000)
cd report/src
npm install
node build_report.js                              # writes report/DeepOCR_Project_Report.docx
powershell -ExecutionPolicy Bypass -File finalize_report.ps1   # fills contents and lists, exports the PDF (needs Word)
```

- `src/content.js` holds the text. Result tables are read from `results/`, so their numbers always match the evaluation files.
- `src/build_report.js` turns the content into Word (A4, Times New Roman 12 pt, 1.5 line spacing, numbered figures and tables).
- `src/finalize_report.ps1` uses Microsoft Word to insert and update the table of contents and the lists of figures and tables.
