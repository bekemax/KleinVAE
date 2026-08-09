"""Execute all code cells in the KleinVAE notebook.

Training is controlled by the notebook's ``KLEINVAE_RUN_TRAINING`` environment
variable. The script does not rewrite notebook outputs; figures and result files
are saved by the notebook cells themselves.
"""

from __future__ import annotations

import argparse
import os
from pathlib import Path

import nbformat


ROOT = Path(__file__).resolve().parents[1]


def execute(notebook_path: Path) -> None:
    os.environ["MPLBACKEND"] = "Agg"
    matplotlib_cache = Path("/tmp/kleinvae-matplotlib")
    matplotlib_cache.mkdir(parents=True, exist_ok=True)
    os.environ["MPLCONFIGDIR"] = str(matplotlib_cache)

    notebook = nbformat.read(notebook_path, as_version=4)
    nbformat.validate(notebook)
    namespace: dict[str, object] = {"__name__": "__notebook_smoke__"}
    for index, cell in enumerate(notebook.cells):
        if cell.cell_type != "code":
            continue
        try:
            exec(compile(cell.source, f"{notebook_path}#cell-{index}", "exec"), namespace)
        except Exception as error:
            raise RuntimeError(
                f"KleinVAE notebook failed at cell {index} ({cell.get('id')})"
            ) from error
        if cell.get("id") == "setup":
            # Avoid verbose dataframe/image representations in a terminal run.
            namespace["display"] = lambda *args, **kwargs: None
        pyplot = namespace.get("plt")
        if pyplot is not None:
            pyplot.close("all")  # type: ignore[union-attr]
        print(f"ok: cell {index} ({cell.get('id')})")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "notebook",
        type=Path,
        nargs="?",
        default=ROOT / "notebooks" / "KleinVAE.ipynb",
    )
    args = parser.parse_args()
    execute(args.notebook)


if __name__ == "__main__":
    main()
