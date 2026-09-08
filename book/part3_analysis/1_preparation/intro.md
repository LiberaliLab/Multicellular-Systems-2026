# Stage 1 — Preparing the data

You will not be handed a tidy table.

The feature table arrives as 733,556 cells × 4,464 columns with **`var` completely empty**
— 4,464 strings and nothing else. The column names carry the imaging channel and round,
but not the antibody, and the only record of which antibody was in which channel is a
separate Excel workbook. Until those two meet, you have 4,464 anonymous numbers.

These three chapters make them meet, then make the result small enough and clean enough to
work with.

```{tableofcontents}
```

By the end you will have an annotated, normalised table that every later chapter opens in
seconds — and, more usefully, a habit of checking what a column *is* before trusting what
it says.

```{important}
Chapter 02 is the memory peak of the whole course. Ask for at least **32 GB** on
JupyterHub — see [Euler and JupyterHub](../../setup/euler_jupyterhub.md).
```
