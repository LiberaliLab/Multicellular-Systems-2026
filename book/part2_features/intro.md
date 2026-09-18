# Part 2 — From images to numbers

Part 1 ended with an OME-Zarr container: pixels, and a label image saying which pixels
belong to which cell. Part 3 begins with a table — **733,556 rows and 4,464 columns**, one
row per cell.

This part is the step in between, and it is the one step of the course you do not run
yourself. The measuring was done before the course started. These two pages explain what it
did, because every column you meet in Part 3 came out of it.

**There is no code in this part.** Read it once; it takes ten minutes.

| | |
|---|---|
| [01](01_what_is_measured.md) | What gets measured — intensity, shape, texture, neighbourhood |
| [02](02_how_and_with_what.md) | How, and with what — regionprops, CellProfiler, Fractal |

:::{tip}
**The one idea.** A segmentation turns pixels into *objects*. Once you have objects, every
measurement has the same shape: pick an object, pick a channel, compute a number. Do that
for every object, every channel and every measure, and you have a table.
:::
