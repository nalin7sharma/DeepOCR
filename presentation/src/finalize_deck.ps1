# Add Morph transitions with Microsoft PowerPoint, save the deck, and export a PDF copy
# plus one PNG per slide (for checking the layout).
#
#   powershell -ExecutionPolicy Bypass -File presentation\src\finalize_deck.ps1
param(
    [string]$Deck = (Join-Path $PSScriptRoot "..\DeepOCR_Presentation.pptx"),
    [string]$ImageDir = "",
    [double]$MorphSeconds = 1.25
)

$ErrorActionPreference = "Stop"
$Deck = (Resolve-Path $Deck).Path
$Pdf = [System.IO.Path]::ChangeExtension($Deck, ".pdf")

$ppEffectMorphByObject = 3954  # writes <p159:morph option="byObject"/>
$ppEffectFadeSmoothly = 3849
$ppSaveAsPDF = 32

$pp = New-Object -ComObject PowerPoint.Application
try {
    # Open(FileName, ReadOnly, Untitled, WithWindow)
    $pres = $pp.Presentations.Open($Deck, 0, 0, 0)
    foreach ($slide in $pres.Slides) {
        $transition = $slide.SlideShowTransition
        if ($slide.SlideIndex -eq 1) {
            $transition.EntryEffect = $ppEffectFadeSmoothly
            $transition.Duration = 0.7
        } else {
            # Morph matches objects by name ("!!lens", "!!card-cnn", ...) and placeholders by type.
            $transition.EntryEffect = $ppEffectMorphByObject
            $transition.Duration = $MorphSeconds
        }
    }
    $pres.Save()
    $pres.SaveAs($Pdf, $ppSaveAsPDF)
    if ($ImageDir) {
        New-Item -ItemType Directory -Force $ImageDir | Out-Null
        foreach ($slide in $pres.Slides) {
            $slide.Export((Join-Path $ImageDir ("slide-{0:D2}.png" -f $slide.SlideIndex)), "PNG", 1600, 900)
        }
    }
    Write-Output "Slides: $($pres.Slides.Count); morph on slides 2-$($pres.Slides.Count)"
    $pres.Close()
    Write-Output "Saved $Deck"
    Write-Output "Exported $Pdf"
} finally {
    $pp.Quit()
    [System.Runtime.InteropServices.Marshal]::ReleaseComObject($pp) | Out-Null
}
