# Fill in the table of contents and the lists of figures and tables with Microsoft Word,
# then save the report and export a PDF copy next to it.
#
#   powershell -ExecutionPolicy Bypass -File report\src\finalize_report.ps1
param(
    [string]$Docx = (Join-Path $PSScriptRoot "..\DeepOCR_Project_Report.docx")
)

$ErrorActionPreference = "Stop"
$Docx = (Resolve-Path $Docx).Path
$Pdf = [System.IO.Path]::ChangeExtension($Docx, ".pdf")

$word = New-Object -ComObject Word.Application
$word.Visible = $false
$word.DisplayAlerts = 0
try {
    $doc = $word.Documents.Open($Docx)

    # Replace a placeholder paragraph such as [[TOC]] with a field.
    function Set-Field($tag, $code) {
        $range = $doc.Content
        $find = $range.Find
        $find.ClearFormatting()
        if ($find.Execute($tag)) {
            $range.Text = ""
            # -1 = wdFieldEmpty: the field code is given as text.
            $null = $doc.Fields.Add($range, -1, $code, $false)
        } else {
            Write-Warning "Placeholder $tag not found"
        }
    }

    Set-Field "[[TOC]]" 'TOC \o "1-2" \h \z'
    Set-Field "[[LOF]]" 'TOC \h \z \t "Figure Caption,1"'
    Set-Field "[[LOT]]" 'TOC \h \z \t "Table Caption,1"'

    # Two passes: inserting the lists shifts the page numbers they report.
    for ($pass = 0; $pass -lt 2; $pass++) {
        foreach ($toc in $doc.TablesOfContents) { $toc.Update() }
        $null = $doc.Fields.Update()
    }

    $doc.Save()
    # 17 = wdExportFormatPDF
    $doc.ExportAsFixedFormat($Pdf, 17)
    Write-Output "Pages: $($doc.ComputeStatistics(2))"
    $doc.Close($false)
    Write-Output "Updated $Docx"
    Write-Output "Exported $Pdf"
} finally {
    $word.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($word) | Out-Null
}
