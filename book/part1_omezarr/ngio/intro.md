# Working with ngio

`ngio` is the fuller library, and the one you will spend most of Part 1 in.

Where `ez-zarr` shows you an image, `ngio` lets you **work** with the whole screen. It
understands wells and acquisitions, reads and writes segmentation labels, handles all four
kinds of table, and works in physical units so you never carry a pixel-to-micrometre
conversion around by hand.

```{tableofcontents}
```

The chapter that matters most is the middle one. `concatenate_image_tables` reads a named
table from every image in a plate and stacks them, adding the well to each row — **384
wells to one table in a single call.**

```{important}
That call is how the Part 3 dataset was made. The 733,556 × 4,464 feature table you decode
in [Stage 1](../../part3_analysis/1_preparation/intro.md) came out of a plate exactly like
the one you work on here. Part 1 is not a warm-up for Part 3; it is the step that produced it.
```

```{note}
This course uses **ngio 1.1.0**. Code written for 0.5.x — including the BioVisionCenter
workshop — differs in a few places. The [cheat sheet](../cheatsheet.md) lists the changes.
```
