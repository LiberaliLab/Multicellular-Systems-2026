# Looking with ez-zarr

`ez-zarr` is the small one. Point it at a plate and get a picture — that is most of what
it does, and it does it well.

Use it when the question is **"what is actually in this well?"**: a quick look at a plate
overview, a channel you are not sure about, a well that produced a suspicious number. It
handles the tedious parts of plotting — channel colours, display ranges, a scale bar in
real micrometres — in a single call.

```{tableofcontents}
```

The chapter ends on the limitation that shapes the rest of Part 1: **`ez-zarr` cannot read
feature tables.** It reads ROI tables happily, but the per-object measurement tables — the
ones the whole of Part 3 is built from — are in a format it does not handle.

That is not a criticism. It is a small library that does looking well, and the next
section introduces the one that does everything else.
