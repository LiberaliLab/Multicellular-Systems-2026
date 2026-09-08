# Part 2 · Exercises

Do these on your laptop with one well open.

## 1. Make a three-channel figure

Display three channels at once, each in its own colour, with contrast limits you chose
deliberately. Turn on the scale bar in micrometres and take a screenshot.

Write down the contrast limits you used. Then set them badly on purpose — much too wide,
then much too narrow — and screenshot each. Keep all three images side by side.

```{admonition} What to notice
:class: dropdown

The three images are of the same data and tell three different stories. Too wide and the
sample looks empty; too narrow and everything looks saturated and uniformly bright.

This is why intensity comparisons belong in the analysis, not the viewer, and why a
figure caption should state its display range. Nothing you did here changed a single
number in the file.
```

## 2. Check a segmentation

Load the label layer for your well and switch it to **contour** mode. Zoom in until you
can see individual cells and find one example of each:

- a mask that merges two cells
- a cell with no mask
- a mask whose boundary is clearly too large or too small

Note the object ids.

```{admonition} What to notice
:class: dropdown

Every one of these becomes a row in the feature table, and nothing downstream marks them
as suspect. A merged object has roughly twice the area and an averaged intensity from two
cells; a boundary that is too loose pulls in background and lowers every mean intensity
measured inside it.

This is what the `is_border_external` flag in
[Part 3 chapter 02](../part3_analysis/02_qc_and_normalisation.ipynb) is doing at scale —
and it only catches cells at a field edge, not these.
```

## 3. From a number back to a picture

Pick an object id you flagged above. In a notebook, using
[Part 1 chapter 3.3](../part1_omezarr/03_ngio/03_labels_tables.ipynb), look up its row in the
feature table. Is it an outlier in area or intensity?

Then do it the other way: find the largest object in the feature table, and go look at it
in napari.

```{admonition} What to notice
:class: dropdown

Being able to go from a number to a picture and back, quickly, is the single most useful
habit in image analysis. An outlier is not a data point — it is a cell, and you can look
at it.

Most segmentation problems are found this way rather than by any automatic metric, which
is why Part 2 sits between the two computational parts rather than at the end.
```

## 4. See the pyramid

Open a well, zoom all the way out, then all the way in on one cell. Watch the status bar
and the sharpness as you zoom.

```{admonition} What to notice
:class: dropdown

Zoomed out you are looking at a coarse pyramid level — a few hundred kilobytes. Zoomed in
you are looking at full resolution, but only for the chunks on screen.

At no point was the whole array read. That is the same mechanism you measured by hand in
[Part 1 chapter 01](../part1_omezarr/01_what_is_omezarr.ipynb), now visible as the reason
a 100 GB plate opens instantly.
```
