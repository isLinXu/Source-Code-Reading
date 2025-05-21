import os
from functools import lru_cache
from pathlib import Path

import datasets


DEFAULT_M4_CACHE_HOME = Path("~/.cache/m4")


@lru_cache(maxsize=1)
def get_m4_cache_dir() -> Path:
    return (Path(os.environ["M4_MANUAL_DIR"]) if "M4_MANUAL_DIR" in os.environ else DEFAULT_M4_CACHE_HOME).expanduser()


@lru_cache(maxsize=1)
def get_jz_dataset_dir() -> Path:
    if "DSDIR" in os.environ:
        return Path(os.environ["DSDIR"]).expanduser()
    raise ValueError("We're not in JZ. This method should only be called when running in JZ.")


# All PMD datasets should following a single feature API.
_FEATURES = datasets.Features(
    {
        "image": datasets.Image(),
        "text": datasets.Value("string"),
        # Define where the sample comes from, this is necessary when we start to use aggregated versions like PMD.
        "source": datasets.Value("string"),
        # We commit any kind of additional information in json format in `meta`
        "meta": datasets.Value("string"),
    }
)
