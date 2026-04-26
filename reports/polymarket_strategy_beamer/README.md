# Polymarket Strategy Beamer Report

This directory contains the Beamer presentation for the Polymarket expert-following backtest.

## Files

- `slides.tex`: LaTeX Beamer source.
- `slides.pdf`: compiled presentation.
- `figures/equity_curve.png`: equity curve figure used by the slides.

## Build

From the repository root:

```bash
pdflatex -interaction=nonstopmode -output-directory reports/polymarket_strategy_beamer reports/polymarket_strategy_beamer/slides.tex
```
