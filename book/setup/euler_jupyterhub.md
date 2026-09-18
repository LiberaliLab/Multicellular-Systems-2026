# Euler and JupyterHub

**Euler** is ETH's central compute cluster. Parts 1 and 3 of this course run there, for
two reasons: the data is far too large to sit on your laptop, and Part 3 needs more
memory than a laptop has.

## 1. Get an account

Every ETH member with a nethz account can use Euler, but the account has to be activated
once. To do this follow to the next steps.

If you are not at ETH, or your account is not active by the first session, tell us
early — this is the one step we cannot fix in the room.

Some extra information can be found at the ETH Scientific Computing instructions at
<https://scicomp.ethz.ch/wiki/Getting_started_with_clusters>.

## 2. Log in from a terminal

You need a terminal once, to create the environment. After that you can work entirely in
the browser.

- **macOS / Linux** — use the built-in **Terminal**.
- **Windows** — use **Windows Terminal** with PowerShell, or **Git Bash**, or WSL.
  Any of them has `ssh`.

```bash
ssh <nethz-username>@euler.ethz.ch
```

Use your nethz password. The first time, you will be asked to confirm the host
fingerprint — type `yes`.

You are now on a **login node**. Login nodes are for editing files and installing
software, not for computation. Everything in this course either runs through JupyterHub
or is small enough not to matter.

### Where your files live

| Path | What for | Notes |
|---|---|---|
| `$HOME` | code, environments, configs | backed up, ~16 GB quota |
| `$SCRATCH` | large temporary files | **deleted after 15 days**, not backed up |
| the course data directory | the plate and feature table | read-only, given to you |

Put the repository and your virtual environment in `$HOME`. Do not copy the dataset
anywhere — you read it where it is.

## 3. Start a notebook server

Go to **<https://jupyter.euler.hpc.ethz.ch>** and log in with your nethz credentials.

You will be asked what resources you want. This matters: the server is a real job on the
cluster, and asking for too little will make Part 3 die halfway through.

| Setting | Parts 1 & 2 | **Part 3** |
|---|---|---|
| Number of cores | 16 | 16 |
| Memory **per core** | 20 GB | **20 GB** |
| Runtime | 4 h | 4 h |
| GPUs | none | none |

```{important}
Part 3 works on a feature table that is **13 GB as a dense float32 array**. Ask for at
least **32 GB in total** (4 cores × 8 GB). With less, Part 3 will be killed without a
useful error message.

Where the memory actually goes is worth knowing, because the answer is not the obvious
one:

| step | peak |
|---|---|
| ch. 01, opening the 13 GB file | **~0.5 GB** — it is opened `backed`, so the array stays on disk |
| ch. 01, writing the slim table | ~8 GB |
| **ch. 02, filtering border cells** | **~16 GB** — the original and the filtered copy exist at once |
| ch. 03–08, one theme at a time | ~8 GB |

So the binding constraint is **chapter 02**, not the chapter that touches the 13 GB file.
Reading a large file costs little if you never materialise it; making a copy of a medium
one costs a lot. 32 GB covers the worst case with room to spare.
```

```{tip}
Do not ask for far more than you need. The larger the request, the longer you wait in the
queue — 4 × 8 GB usually starts in a minute or two.
```

Asking for more than you need is not free: the larger the request, the longer you wait
in the queue. 4 × 16 GB usually starts within a minute or two.

