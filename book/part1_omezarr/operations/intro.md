# Working with OME-Zarr

Two libraries, used for different things. Learning when to reach for which is most of the
skill.

**`ez-zarr`** is small and immediate. Point it at a plate, get a picture. It is the right
tool for *looking* — a quick check that a well contains what you think it does.

**`ngio`** is the fuller library. It understands wells and acquisitions, reads and writes
labels and tables, and can pull one measurement table out of all 384 wells in a single
call. That last capability is how the Part 3 dataset was built, so this is where Part 1
connects to the rest of the course.

```{tableofcontents}
```

```{note}
Both are installed by `environment/requirements.txt` and coexist happily — `ez-zarr`
needs `zarr>=3.0`, `ngio` needs `zarr>=3.1.6`, and the second satisfies the first. You do
not need separate environments for them.
```
