"""Validate and visualize smoothness-feature behavior in the breast cancer dataset."""

from __future__ import annotations

import argparse
import csv
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np


LABEL_COLORS = {
    0: "#93c5fd",  # benign
    1: "#fde68a",  # malignant
}

LABEL_NAMES = {
    0: "Benign",
    1: "Malignant",
}

FEATURES = [
    ("smoothness1", "Smoothness 1\n(Mean)"),
    ("smoothness2", "Smoothness 2\n(SE)"),
    ("smoothness3", "Smoothness 3\n(Worst)"),
]

FEATURE_COLORS = {
    "smoothness1": "#0f766e",
    "smoothness2": "#c2410c",
    "smoothness3": "#6d28d9",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze smoothness-feature behavior for the breast cancer WDBC example."
    )
    parser.add_argument(
        "--csv",
        default="backend/datasets/examples/breast_cancer/breast_cancer_wdbc.csv",
        help="Path to the normalized breast cancer CSV.",
    )
    parser.add_argument(
        "--tmap",
        default="backend/datasets/examples/breast_cancer/breast_cancer_wdbc.tmap",
        help="Path to the breast cancer .tmap file.",
    )
    parser.add_argument(
        "--output-dir",
        default="output/paper_figures/breast_cancer_smoothness",
        help="Directory for generated figure files.",
    )
    return parser.parse_args()


def load_csv_rows(csv_path: Path) -> list[dict[str, str]]:
    with csv_path.open() as handle:
        return list(csv.DictReader(handle))


def load_tmap_payload(tmap_path: Path) -> dict:
    return json.loads(tmap_path.read_text())


def auc_rank(values: np.ndarray, labels: np.ndarray) -> float:
    pairs = sorted(zip(values.tolist(), labels.tolist()), key=lambda item: item[0])
    n_pos = int(np.sum(labels == 1))
    n_neg = int(np.sum(labels == 0))
    rank = 1
    idx = 0
    rank_sum_pos = 0.0
    while idx < len(pairs):
        end = idx
        while end < len(pairs) and pairs[end][0] == pairs[idx][0]:
            end += 1
        avg_rank = (rank + (rank + (end - idx) - 1)) / 2.0
        rank_sum_pos += avg_rank * sum(label for _, label in pairs[idx:end])
        rank += end - idx
        idx = end
    return (rank_sum_pos - n_pos * (n_pos + 1) / 2.0) / max(n_pos * n_neg, 1)


def cohens_d(group_a: np.ndarray, group_b: np.ndarray) -> float:
    mean_a = float(np.mean(group_a))
    mean_b = float(np.mean(group_b))
    var_a = float(np.var(group_a, ddof=1))
    var_b = float(np.var(group_b, ddof=1))
    pooled = math.sqrt(
        (((len(group_a) - 1) * var_a) + ((len(group_b) - 1) * var_b))
        / max(len(group_a) + len(group_b) - 2, 1)
    )
    return (mean_b - mean_a) / max(pooled, 1e-9)


def fit_logistic_standardized(
    matrix: np.ndarray,
    labels: np.ndarray,
    steps: int = 4000,
    learning_rate: float = 0.1,
) -> np.ndarray:
    means = matrix.mean(axis=0)
    stds = matrix.std(axis=0)
    stds[stds <= 0] = 1.0
    z = (matrix - means) / stds

    weights = np.zeros(z.shape[1], dtype=float)
    bias = 0.0
    n = z.shape[0]
    for _ in range(steps):
        grad_w = np.zeros_like(weights)
        grad_b = 0.0
        for row, label in zip(z, labels):
            score = bias + float(np.dot(weights, row))
            if score >= 0:
                exp_term = math.exp(-score)
                prob = 1.0 / (1.0 + exp_term)
            else:
                exp_term = math.exp(score)
                prob = exp_term / (1.0 + exp_term)
            diff = prob - float(label)
            grad_w += diff * row
            grad_b += diff
        weights -= learning_rate * grad_w / n
        bias -= learning_rate * grad_b / n
    return weights


def choose_representative_points(positions: np.ndarray, labels: np.ndarray, axis: np.ndarray) -> dict[str, int]:
    benign_idx = np.where(labels == 0)[0]
    malignant_idx = np.where(labels == 1)[0]
    benign_centroid = positions[benign_idx].mean(axis=0)
    malignant_centroid = positions[malignant_idx].mean(axis=0)
    midpoint = (benign_centroid + malignant_centroid) / 2.0

    benign_core = benign_idx[np.argmin(np.linalg.norm(positions[benign_idx] - benign_centroid, axis=1))]
    malignant_core = malignant_idx[np.argmin(np.linalg.norm(positions[malignant_idx] - malignant_centroid, axis=1))]
    benign_boundary = benign_idx[np.argmax((positions[benign_idx] - midpoint) @ axis)]

    return {
        "benign_core": int(benign_core),
        "benign_boundary": int(benign_boundary),
        "malignant_core": int(malignant_core),
    }


def draw_grouped_boxplots(ax, csv_values: dict[str, dict[int, np.ndarray]], metrics: dict[str, dict[str, float]]) -> None:
    centers = np.arange(len(FEATURES), dtype=float)
    width = 0.28
    for offset_idx, label in enumerate((0, 1)):
        offset = (offset_idx - 0.5) * 0.34
        values = [csv_values[feature][label] for feature, _ in FEATURES]
        bp = ax.boxplot(
            values,
            positions=centers + offset,
            widths=width,
            patch_artist=True,
            showfliers=False,
            medianprops={"color": "#1f2937", "linewidth": 1.7},
            whiskerprops={"color": "#6b7280", "linewidth": 1.1},
            capprops={"color": "#6b7280", "linewidth": 1.1},
            boxprops={"linewidth": 1.1},
        )
        for patch in bp["boxes"]:
            patch.set_facecolor(LABEL_COLORS[label])
            patch.set_edgecolor("#475569")
            patch.set_alpha(0.9)

    ax.set_xticks(centers)
    ax.set_xticklabels([display for _, display in FEATURES], fontsize=10)
    ax.set_ylabel("Normalized Feature Value", fontsize=11)
    ax.set_title("Raw Label Separation", fontsize=13, fontweight="bold", color="#1f1b17")
    ax.grid(axis="y", color="#e5ded1", linewidth=0.75, alpha=0.7)
    ax.set_facecolor("#fbfaf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for idx, (feature, _) in enumerate(FEATURES):
        d_value = metrics[feature]["cohens_d"]
        ax.text(
            centers[idx],
            1.03,
            f"d={d_value:+.2f}",
            ha="center",
            va="bottom",
            fontsize=9.5,
            color="#5b5144",
        )


def draw_effect_summary(ax, metrics: dict[str, dict[str, float]]) -> None:
    centers = np.arange(len(FEATURES), dtype=float)
    width = 0.34
    cohens = [metrics[feature]["cohens_d"] for feature, _ in FEATURES]
    logistic = [metrics[feature]["logistic_coef"] for feature, _ in FEATURES]
    ax.bar(
        centers - width / 2.0,
        cohens,
        width=width,
        color="#475569",
        alpha=0.9,
        label="Cohen's d\n(Malignant - Benign)",
    )
    ax.bar(
        centers + width / 2.0,
        logistic,
        width=width,
        color="#d97706",
        alpha=0.9,
        label="Standardized\nLogistic Coef.",
    )
    ax.axhline(0.0, color="#64748b", linewidth=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([display for _, display in FEATURES], fontsize=10)
    ax.set_ylabel("Signed Effect Size", fontsize=11)
    ax.set_title("Raw-Data Effect Summary", fontsize=13, fontweight="bold", color="#1f1b17")
    ax.grid(axis="y", color="#e5ded1", linewidth=0.75, alpha=0.7)
    ax.set_facecolor("#fbfaf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def draw_tmap_projection_summary(ax, metrics: dict[str, dict[str, float]]) -> None:
    centers = np.arange(len(FEATURES), dtype=float)
    width = 0.34
    benign_signed = [metrics[feature]["benign_signed_projection"] for feature, _ in FEATURES]
    malignant_signed = [metrics[feature]["malignant_signed_projection"] for feature, _ in FEATURES]
    ax.bar(
        centers - width / 2.0,
        benign_signed,
        width=width,
        color=LABEL_COLORS[0],
        edgecolor="#475569",
        linewidth=0.8,
        label="Mean Projection on Benign Side",
    )
    ax.bar(
        centers + width / 2.0,
        malignant_signed,
        width=width,
        color=LABEL_COLORS[1],
        edgecolor="#475569",
        linewidth=0.8,
        label="Mean Projection on Malignant Side",
    )
    ax.axhline(0.0, color="#64748b", linewidth=1.0)
    ax.set_xticks(centers)
    ax.set_xticklabels([display for _, display in FEATURES], fontsize=10)
    ax.set_ylabel("Projection onto Benign→Malignant Axis", fontsize=11)
    ax.set_title("Map Direction Summary", fontsize=13, fontweight="bold", color="#1f1b17")
    ax.grid(axis="y", color="#e5ded1", linewidth=0.75, alpha=0.7)
    ax.set_facecolor("#fbfaf7")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.legend(frameon=False, fontsize=9, loc="upper left")


def draw_tmap_panel(
    ax,
    positions: np.ndarray,
    labels: np.ndarray,
    benign_centroid: np.ndarray,
    malignant_centroid: np.ndarray,
    mean_vectors: dict[str, dict[int, np.ndarray]],
    representative_points: dict[str, int],
) -> None:
    for label in (0, 1):
        mask = labels == label
        ax.scatter(
            positions[mask, 0],
            positions[mask, 1],
            s=22,
            color=LABEL_COLORS[label],
            edgecolors="white",
            linewidths=0.35,
            alpha=0.82,
            zorder=2,
        )

    ax.scatter(
        [benign_centroid[0], malignant_centroid[0]],
        [benign_centroid[1], malignant_centroid[1]],
        s=100,
        c=["#2563eb", "#ca8a04"],
        edgecolors="#111827",
        linewidths=0.8,
        zorder=5,
    )
    ax.annotate(
        "",
        xy=(malignant_centroid[0], malignant_centroid[1]),
        xytext=(benign_centroid[0], benign_centroid[1]),
        arrowprops={"arrowstyle": "->", "linewidth": 1.5, "color": "#64748b"},
        zorder=4,
    )
    ax.text(
        benign_centroid[0],
        benign_centroid[1] + 1.8,
        "Benign\ncentroid",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f2937",
    )
    ax.text(
        malignant_centroid[0],
        malignant_centroid[1] + 1.8,
        "Malignant\ncentroid",
        ha="center",
        va="bottom",
        fontsize=9,
        color="#1f2937",
    )

    scale = 10.0
    for feature, _display in FEATURES:
        color = FEATURE_COLORS[feature]
        benign_vec = mean_vectors[feature][0] * scale
        malignant_vec = mean_vectors[feature][1] * scale
        ax.arrow(
            benign_centroid[0],
            benign_centroid[1],
            benign_vec[0],
            benign_vec[1],
            color=color,
            width=0.18,
            head_width=1.5,
            head_length=2.0,
            length_includes_head=True,
            alpha=0.92,
            zorder=6,
        )
        ax.arrow(
            malignant_centroid[0],
            malignant_centroid[1],
            malignant_vec[0],
            malignant_vec[1],
            color=color,
            width=0.12,
            head_width=1.3,
            head_length=1.7,
            length_includes_head=True,
            alpha=0.6,
            linestyle="--",
            zorder=6,
        )

    marker_styles = {
        "benign_core": ("o", "#1d4ed8", "Benign core"),
        "benign_boundary": ("s", "#7c3aed", "Benign boundary"),
        "malignant_core": ("D", "#a16207", "Malignant core"),
    }
    for key, idx in representative_points.items():
        marker, color, label = marker_styles[key]
        ax.scatter(
            positions[idx, 0],
            positions[idx, 1],
            s=72,
            marker=marker,
            facecolors="none",
            edgecolors=color,
            linewidths=1.8,
            zorder=7,
        )
        ax.text(
            positions[idx, 0] + 1.3,
            positions[idx, 1] + 1.0,
            label,
            fontsize=8.5,
            color=color,
            zorder=8,
        )

    handles = [
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[0], markeredgecolor="white", markersize=8, label="Benign points"),
        Line2D([0], [0], marker="o", color="w", markerfacecolor=LABEL_COLORS[1], markeredgecolor="white", markersize=8, label="Malignant points"),
    ]
    handles.extend(
        Line2D([0], [0], color=FEATURE_COLORS[feature], linewidth=2.4, label=display.replace("\n", " "))
        for feature, display in FEATURES
    )
    ax.legend(handles=handles, frameon=False, fontsize=8.5, loc="upper right")
    ax.set_title("Embedding-Level Direction Check", fontsize=13, fontweight="bold", color="#1f1b17")
    ax.set_xlabel("t-SNE 1", fontsize=11)
    ax.set_ylabel("t-SNE 2", fontsize=11)
    ax.set_facecolor("#fbfaf7")
    ax.grid(color="#e5ded1", linewidth=0.75, alpha=0.45)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)


def write_summary_markdown(
    output_path: Path,
    metrics: dict[str, dict[str, float]],
    representative_points: dict[str, int],
) -> None:
    lines = [
        "# Breast Cancer Smoothness Analysis",
        "",
        "Label mapping: `0 = benign`, `1 = malignant`.",
        "",
        "Interpretation supported by the data:",
        "",
        "- `smoothness1` (mean) and `smoothness3` (worst) are positive malignant markers.",
        "- `smoothness2` (standard error) is a weaker and slightly opposing signal.",
        "- In the `.tmap`, the benign-side mean vector for `smoothness1` and `smoothness3` points toward the malignant centroid, while `smoothness2` points back toward the benign side.",
        "- The malignant-side mean vectors are weaker, which is consistent with a saturation effect once a point is already inside the malignant region.",
        "",
        "## Metrics",
        "",
        "| Feature | Benign Mean | Malignant Mean | AUC for Malignant | Cohen's d | Logistic Coef. | Benign Axis Projection | Malignant Axis Projection |",
        "| --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for feature, display in FEATURES:
        data = metrics[feature]
        lines.append(
            f"| {display.replace(chr(10), ' ')} | "
            f"{data['benign_mean']:.3f} | {data['malignant_mean']:.3f} | "
            f"{data['auc']:.3f} | {data['cohens_d']:+.3f} | {data['logistic_coef']:+.3f} | "
            f"{data['benign_signed_projection']:+.3f} | {data['malignant_signed_projection']:+.3f} |"
        )
    lines.extend(
        [
            "",
            "## Representative TMAP Points",
            "",
            f"- Benign core point index: `{representative_points['benign_core']}`",
            f"- Benign boundary point index: `{representative_points['benign_boundary']}`",
            f"- Malignant core point index: `{representative_points['malignant_core']}`",
            "",
            "These point markers are shown in the embedding panel for visual reference only; the validation claim is based on the full-dataset statistics above, not on those three points alone.",
        ]
    )
    output_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    args = parse_args()
    csv_path = Path(args.csv)
    tmap_path = Path(args.tmap)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    rows = load_csv_rows(csv_path)
    tmap_payload = load_tmap_payload(tmap_path)
    col_labels = list(tmap_payload["Col_labels"])

    csv_labels = np.asarray([int(row["label"]) for row in rows], dtype=int)
    csv_values = {
        feature: {
            0: np.asarray([float(row[feature]) for row in rows if int(row["label"]) == 0], dtype=float),
            1: np.asarray([float(row[feature]) for row in rows if int(row["label"]) == 1], dtype=float),
        }
        for feature, _display in FEATURES
    }
    csv_feature_all = {
        feature: np.asarray([float(row[feature]) for row in rows], dtype=float)
        for feature, _display in FEATURES
    }

    feature_matrix = np.asarray(
        [[float(row[feature]) for feature, _display in FEATURES] for row in rows],
        dtype=float,
    )
    logistic_weights = fit_logistic_standardized(feature_matrix, csv_labels)

    positions = np.asarray([entry["range"][:2] for entry in tmap_payload["tmap"]], dtype=float)
    tmap_labels = np.asarray([int(entry["class"]) for entry in tmap_payload["tmap"]], dtype=int)
    gradients = np.asarray([np.asarray(entry["tangent"], dtype=float) for entry in tmap_payload["tmap"]], dtype=float)

    benign_centroid = positions[tmap_labels == 0].mean(axis=0)
    malignant_centroid = positions[tmap_labels == 1].mean(axis=0)
    malignant_axis = malignant_centroid - benign_centroid
    malignant_axis = malignant_axis / np.linalg.norm(malignant_axis)

    metrics: dict[str, dict[str, float]] = {}
    mean_vectors: dict[str, dict[int, np.ndarray]] = {}
    for feature_idx, (feature, _display) in enumerate(FEATURES):
        benign_values = csv_values[feature][0]
        malignant_values = csv_values[feature][1]
        tmap_feature_idx = col_labels.index(feature)
        feature_gradients = gradients[:, :, tmap_feature_idx]
        signed_projection = feature_gradients @ malignant_axis

        metrics[feature] = {
            "benign_mean": float(np.mean(benign_values)),
            "malignant_mean": float(np.mean(malignant_values)),
            "auc": float(auc_rank(csv_feature_all[feature], csv_labels)),
            "cohens_d": float(cohens_d(benign_values, malignant_values)),
            "logistic_coef": float(logistic_weights[feature_idx]),
            "benign_signed_projection": float(np.mean(signed_projection[tmap_labels == 0])),
            "malignant_signed_projection": float(np.mean(signed_projection[tmap_labels == 1])),
        }
        mean_vectors[feature] = {
            0: feature_gradients[tmap_labels == 0].mean(axis=0),
            1: feature_gradients[tmap_labels == 1].mean(axis=0),
        }

    representative_points = choose_representative_points(positions, tmap_labels, malignant_axis)

    fig, axes = plt.subplots(2, 2, figsize=(15.6, 11.8))
    fig.patch.set_facecolor("white")
    draw_grouped_boxplots(axes[0, 0], csv_values=csv_values, metrics=metrics)
    draw_effect_summary(axes[0, 1], metrics=metrics)
    draw_tmap_projection_summary(axes[1, 0], metrics=metrics)
    draw_tmap_panel(
        axes[1, 1],
        positions=positions,
        labels=tmap_labels,
        benign_centroid=benign_centroid,
        malignant_centroid=malignant_centroid,
        mean_vectors=mean_vectors,
        representative_points=representative_points,
    )
    fig.suptitle(
        "Breast Cancer WDBC: Why Smoothness 1 and 3 Push Toward Malignancy While Smoothness 2 Pulls Back",
        fontsize=17,
        fontweight="bold",
        color="#1f1b17",
        y=0.985,
    )
    fig.text(
        0.5,
        0.958,
        "Validation combines raw label separation with embedding-level direction along the benign→malignant axis.",
        ha="center",
        va="top",
        fontsize=10.8,
        color="#5d554d",
    )
    fig.tight_layout(rect=[0.03, 0.03, 0.99, 0.94])

    png_path = output_dir / "breast_cancer_smoothness_validation.png"
    pdf_path = output_dir / "breast_cancer_smoothness_validation.pdf"
    fig.savefig(png_path, dpi=220, bbox_inches="tight")
    fig.savefig(pdf_path, dpi=300, bbox_inches="tight")
    plt.close(fig)

    summary_path = output_dir / "breast_cancer_smoothness_analysis.md"
    write_summary_markdown(summary_path, metrics=metrics, representative_points=representative_points)

    for path in (png_path, pdf_path, summary_path):
        print(path)


if __name__ == "__main__":
    main()
