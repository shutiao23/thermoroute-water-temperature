# Editable ThermoRoute paper figures

## What the previous long preview was

The previous long image was a **contact sheet**: three independent figures were
stacked vertically only so that you could compare them. It was not intended to
be inserted into the paper as one giant figure.

## Recommended diagram count

### Main text: two method diagrams

1. `fig_main_model_concept`
   - explains the common anchor + residual formulation;
   - deliberately does not show every neural layer.

2. `fig_information_regime_framework`
   - the most important design figure;
   - defines F, L, G, and A;
   - shows the 48-cell crossed experiment and the paired contrasts.

### Supporting Information: one architecture diagram

3. `figS_full_thermoroute_architecture`
   - complete ThermoRoute dataflow;
   - inputs, anchor, proposal, selector, TCN, experts, heads, calibration;
   - includes the interpretation limits.

For the full manuscript, these are only the method diagrams. A compact final
main-text plan is usually five figures:

1. sites + temporal partitions;
2. minimal model concept;
3. F x L x G x A identification framework;
4. forcing / crossed-response results;
5. hydrologic heterogeneity and event relevance.

If the publication-unit limit is tight, merge the site/timeline panel and the
minimal model concept, leaving four or five main figures.

## Install

```bash
python -m venv .venv
source .venv/bin/activate       # Linux/macOS
# .venv\Scripts\activate        # Windows PowerShell

python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

## Draw all figures

```bash
python draw_thermoroute_figures.py \
  --figure all \
  --outdir generated_figures
```

## Draw one figure

```bash
python draw_thermoroute_figures.py \
  --figure framework \
  --outdir generated_figures
```

Allowed names:

```text
concept
framework
architecture
all
```

## Outputs

Each figure is exported as:

- SVG: fully editable in Inkscape or Illustrator;
- PDF: vector file for LaTeX;
- PNG: quick preview.

## Where to edit

Edit the semantic colors at the top:

```python
C = {...}
```

Edit positions and wording in:

```python
draw_main_model_concept(...)
draw_information_regime_framework(...)
draw_full_architecture(...)
```

The SVG coordinate system uses explicit pixel coordinates. Typical calls are:

```python
rounded_rect(dwg, x, y, width, height, ...)
text(dwg, "label", x, y, ...)
path(dwg, "M ... C ...", ...)
```

## Compile the demonstration TeX file

First generate figures, then:

```bash
pdflatex figure_embed_example.tex
pdflatex figure_embed_example.tex
```

or:

```bash
latexmk -pdf figure_embed_example.tex
```

## Insert into an AGU manuscript

Use the generated PDF, not PNG:

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[
    width=\linewidth,
    height=0.47\textheight,
    keepaspectratio
  ]{generated_figures/fig_information_regime_framework.pdf}
  \caption{Experimental identification of river-temperature information
  regimes. ...}
  \label{fig:information-regimes}
\end{figure}
```

For the SI figure:

```latex
\begin{figure}[!t]
  \centering
  \includegraphics[
    width=\linewidth,
    height=0.48\textheight,
    keepaspectratio
  ]{generated_figures/figS_full_thermoroute_architecture.pdf}
  \caption{Full ThermoRoute architecture and calibration dataflow. ...}
  \label{fig:si-architecture}
\end{figure}
```

Avoid `[H]` unless necessary. `[!t]` gives a more natural paper layout.
