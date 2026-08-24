"""The single definition of this SDK's version.

Every recorded package stamps this into its manifest, which is the only way a
bug report can be tied back to the build that produced the data. It was
previously written out in three places -- `pyproject.toml`, `__init__` and the
manifest's `SDK` block -- so they drifted, and every package ever recorded
claimed `0.1.0` however much had changed underneath it.

`pyproject.toml` reads this file, so the packaged version and the reported
version cannot disagree.
"""

__version__ = "0.5.1"
