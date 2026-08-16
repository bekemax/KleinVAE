"""Generate the paper's appendix figures from exported W&B curves."""

from __future__ import annotations

import argparse
import base64
import hashlib
import json
import subprocess
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "data" / "appendix"

MODEL_COLUMNS = {
    "Vanilla VAE 2D": "vanilla_vae_2D",
    "Vanilla VAE 3D": "vanilla_vae_3D",
    "Vanilla VAE 4D": "vanilla_vae_4D",
    "Torus VAE": "torus_vae",
    "Klein VAE": "klein_vae",
}
LINESTYLES = ["--", "-.", ":", (0, (3, 1, 1, 1)), "-"]
MARKERS = ["o", "s", "^", "v", "X"]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _git_blob(path: Path) -> str | None:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", f"a731985:{path.relative_to(ROOT)}"],
            cwd=ROOT,
            text=True,
            stderr=subprocess.DEVNULL,
        ).strip()
    except (OSError, subprocess.CalledProcessError):
        return None


def _notebook_metadata() -> dict:
    """Identify the committed plotting code and its embedded figure outputs."""

    notebook_path = Path("notebooks/iclr-reports.ipynb")
    raw = subprocess.check_output(
        ["git", "show", f"a731985:{notebook_path}"], cwd=ROOT
    )
    notebook = json.loads(raw)
    embedded_outputs: list[dict[str, str | int]] = []
    referenced_csvs: list[str] = []
    for cell_index, cell in enumerate(notebook.get("cells", [])):
        source = "".join(cell.get("source", []))
        referenced_csvs.extend(
            str(path.relative_to(ROOT))
            for path in (
                SOURCE_DIR / "elbo.csv",
                SOURCE_DIR / "var_z.csv",
                SOURCE_DIR / "bottleneck_2.csv",
                SOURCE_DIR / "bottleneck_3.csv",
            )
            if str(path.relative_to(ROOT)) in source
        )
        for output_index, output in enumerate(cell.get("outputs", [])):
            encoded = output.get("data", {}).get("image/png")
            if encoded is None:
                continue
            image_bytes = base64.b64decode(encoded)
            embedded_outputs.append(
                {
                    "cell": cell_index,
                    "output": output_index,
                    "sha256": hashlib.sha256(image_bytes).hexdigest(),
                }
            )
    return {
        "path": str(notebook_path),
        "commit": "a731985",
        "git_blob": subprocess.check_output(
            ["git", "rev-parse", f"a731985:{notebook_path}"],
            cwd=ROOT,
            text=True,
        ).strip(),
        "sha256": hashlib.sha256(raw).hexdigest(),
        "referenced_local_csv_paths": sorted(set(referenced_csvs)),
        "embedded_png_outputs": embedded_outputs,
    }


def _plot_series(
    axis: plt.Axes,
    frame: pd.DataFrame,
    suffix: str,
    *,
    logarithmic: bool = False,
) -> None:
    for (label, prefix), linestyle, marker in zip(
        MODEL_COLUMNS.items(), LINESTYLES, MARKERS, strict=True
    ):
        column = f"{prefix} - {suffix}"
        if column not in frame:
            raise KeyError(f"Missing data column: {column}")
        axis.plot(
            frame["epoch"],
            frame[column],
            color="black",
            linestyle=linestyle,
            marker=marker,
            markersize=5,
            markevery=max(1, len(frame) // 10),
            label=label,
        )
    if logarithmic:
        axis.set_yscale("log", base=10)
    axis.set_xlabel("epoch")
    axis.grid(True, which="both", linestyle="--", linewidth=0.5)
    axis.legend(fontsize=8)


def _summaries(frames: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows: list[dict[str, float | str]] = []
    specs = {
        "optimized_validation_loss": (frames["elbo"], "val_loss"),
        "legacy_var_z": (frames["var_z"], "var_z"),
        "bottleneck_z2": (frames["bottleneck_2"], "total_bottleneck_over_2"),
        "bottleneck_z3": (frames["bottleneck_3"], "total_bottleneck_over_3"),
    }
    for metric, (frame, suffix) in specs.items():
        for label, prefix in MODEL_COLUMNS.items():
            values = frame[f"{prefix} - {suffix}"].dropna()
            rows.append(
                {
                    "metric": metric,
                    "model": label,
                    "first": float(values.iloc[0]),
                    "final": float(values.iloc[-1]),
                    "minimum": float(values.min()),
                    "maximum": float(values.max()),
                    "logged_points": int(len(values)),
                }
            )
    return pd.DataFrame(rows)


def generate_figures(output_dir: Path) -> dict[str, Path]:
    output_dir.mkdir(parents=True, exist_ok=True)
    source_paths = {
        name: SOURCE_DIR / f"{name}.csv"
        for name in ("elbo", "var_z", "bottleneck_2", "bottleneck_3")
    }
    frames = {name: pd.read_csv(path) for name, path in source_paths.items()}

    figure5, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _plot_series(axes[0], frames["elbo"], "val_loss")
    axes[0].set_title("Evidence Lower Bound (ELBO)")
    axes[0].set_ylabel("ELBO")
    _plot_series(axes[1], frames["var_z"], "var_z", logarithmic=True)
    axes[1].set_title("Variance of Latent Codes")
    axes[1].set_ylabel(r"$\log_{10}(\mathrm{Var}(z))$")
    figure5.tight_layout()
    figure5_path = output_dir / "figure5.png"
    figure5.savefig(figure5_path, dpi=300, bbox_inches="tight")
    plt.close(figure5)

    figure6, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    _plot_series(axes[0], frames["bottleneck_2"], "total_bottleneck_over_2")
    axes[0].set_title(r"Norm of Bottleneck Distance over $\mathbb{Z}_2$")
    axes[0].set_ylabel(r"$\|d_B\|_2$")
    _plot_series(axes[1], frames["bottleneck_3"], "total_bottleneck_over_3")
    axes[1].set_title(r"Norm of Bottleneck Distance over $\mathbb{Z}_3$")
    axes[1].set_ylabel(r"$\|d_B\|_2$")
    figure6.tight_layout()
    figure6_path = output_dir / "figure6.png"
    figure6.savefig(figure6_path, dpi=300, bbox_inches="tight")
    plt.close(figure6)

    summary_path = output_dir / "curve_summary.csv"
    _summaries(frames).to_csv(summary_path, index=False)

    metadata = {
        "source_commit": "a731985",
        "plot_notebook": _notebook_metadata(),
        "figure_data": {
            name: {
                "path": str(path.relative_to(ROOT)),
                "sha256": _sha256(path),
                "git_blob_at_a731985": _git_blob(path),
                "tracked_at_a731985": _git_blob(path) is not None,
                "source": "W&B export referenced by the plotting notebook.",
                "rows": len(frames[name]),
            }
            for name, path in source_paths.items()
        },
        "metric_semantics": {
            "figure_5a": (
                "Exported val_loss. Klein/Torus used pixel-mean BCE; Euclidean "
                "2D/3D/4D used pixel-summed MSE divided by batch size. This is "
                "not a comparable ELBO across models."
            ),
            "figure_5b": (
                "One-sample empirical sum of coordinate variances on the fixed "
                "500-point validation subset. Klein/Torus samples were wrapped "
                "with project_to_torus; Euclidean samples were unprojected."
            ),
            "figure_6": (
                "L2 norm across H0/H1/H2 bottleneck distances between ambient "
                "persistence diagrams of 500 validation images and stochastic "
                "reconstructions."
            ),
        },
        "configuration_notes": {
            "dataset_size": "Figure data use 10,000 images; the proof experiment uses 100,000.",
            "batch_size": "Figure data use batch size 512; the proof experiment uses 1,024.",
            "scheduler": "Torus/Euclidean runs use factor 0.1 and patience 10.",
            "seed": "The run configuration does not record a seed.",
        },
        "data_note": (
            "Commit a731985 contains the plotting code and embedded PNG outputs. "
            "The CSV exports were not tracked in that commit; their current "
            "SHA-256 hashes are recorded above."
        ),
    }
    metadata_path = output_dir / "metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2) + "\n")

    return {
        "figure5": figure5_path,
        "figure6": figure6_path,
        "summary": summary_path,
        "metadata": metadata_path,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=ROOT / "results" / "appendix",
    )
    args = parser.parse_args()
    for name, path in generate_figures(args.output_dir).items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
