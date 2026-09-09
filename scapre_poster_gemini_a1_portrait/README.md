# Informax Channel Weighting in ScaPre (A1 Portrait)

This folder contains an Overleaf-ready A1 portrait LaTeX poster. It uses a traditional two-column layout that is read from the top to the bottom of the left column, then from the top to the bottom of the right column.

## Main file
- `poster.tex`

## Included assets
- `figures/` - report figures and the vector two-panel saturation figure

The two-panel figure can be regenerated without Matplotlib after activating the appropriate project environment:

```bash
python figures/build_mi_saturation_panels.py
```

## Compile
On Overleaf, use `pdflatex` (or `latexmk -pdf`) twice.

Example:

```bash
pdflatex poster.tex
pdflatex poster.tex
```

For a lightweight local setup on macOS, install and run Tectonic:

```bash
brew install tectonic
tectonic poster.tex --keep-logs
```

## Notes
- The header intentionally omits institutional seals and logos.
- The poster includes the ScaPre paper citation and no QR-code block.
- The layout is A1 portrait.
- The title is 49 pt, section headings are 34 pt, and body text is 24 pt.
- Tables are 22 pt, figure labels are at least 18 pt after placement, and the reference is 20 pt.
- Names are formatted consistently as:
  - Tse-An Li
  - Te-Sheng Lin
  - Jia-Wei Liao
