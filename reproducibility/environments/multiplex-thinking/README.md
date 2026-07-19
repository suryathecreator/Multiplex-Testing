# `multiplex-thinking` environment lock

This directory freezes the home Conda environment that was removed during the
July 2026 quota cleanup. It is intentionally committed before deletion.

Recreate the Conda base exactly with:

```bash
conda create -p /path/to/multiplex-thinking --file conda-explicit.lock
```

Then install the pip snapshot from `pip-freeze.lock`. The two editable packages
must come from the Multiplex repository and commit recorded in
`source-revisions.json`; the absolute conda-build paths for `packaging` and
`pip` in the freeze are informational because those packages are already
pinned by the explicit Conda lock.

`conda-history.lock` records the original creation command. These files do not
claim cross-platform portability; they target the original `linux-64` system.
