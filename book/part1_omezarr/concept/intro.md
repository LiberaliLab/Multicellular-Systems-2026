# Understanding OME-Zarr

Before reaching for a library, it is worth knowing what you are actually opening.

An OME-Zarr is not a file. It is a **directory tree of small files**, holding the same
image at several resolutions, with its metadata in plain JSON beside the pixels. Every
convenience that `ez-zarr` and `ngio` offer in the next section is built on that, and
their behaviour makes far more sense once you have seen the raw thing.

```{tableofcontents}
```

One chapter, and it uses no imaging library at all — just `pathlib` and `json`. By the end
you will be able to open a plate in a file browser and say what every directory is for.
