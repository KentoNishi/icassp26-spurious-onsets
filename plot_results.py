import argparse
import colorsys
import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.offsetbox import AnchoredOffsetbox, HPacker, TextArea


MOSHI = "#d62728"
PERSONAPLEX = "#76b900"
MOSHI_TRACE = "#e78ac3"
PERSONAPLEX_TRACE = "#2ab7a9"
GRID = "#dddddd"
TEXT = "#333333"
OUT = Path(__file__).parent / "figures"
WIDTH = 86 / 25.4 * 72  # PDF points: match the manuscript's column width.
FONT_SIZE = 7


def paper_axes(height, boxes):
    """Place plot interiors in the manuscript's final, top-origin PDF geometry."""
    plt.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["DejaVu Sans"],
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "axes.labelpad": 5,
        }
    )
    fig = plt.figure(figsize=(WIDTH / 72, height / 72))
    axes = [
        fig.add_axes(
            (
                left / WIDTH,
                (height - bottom) / height,
                (right - left) / WIDTH,
                (bottom - top) / height,
            )
        )
        for left, top, right, bottom in boxes
    ]
    return fig, axes


def model_title(fig, ax, name, height):
    fig.text(
        (ax.get_position().x0 + ax.get_position().x1) / 2,
        1 - 10 / height,
        name,
        ha="center",
        va="baseline",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=TEXT,
    )


def style(ax, *, grid_axis="both"):
    for spine in ax.spines.values():
        spine.set_color(TEXT)
        spine.set_linewidth(0.65)
    ax.tick_params(colors=TEXT, labelsize=FONT_SIZE, width=0.65, length=2, pad=2)
    if grid_axis:
        ax.grid(axis=grid_axis, color=GRID, linewidth=0.5)
    ax.set_axisbelow(True)


def shades(color, count):
    hue, lightness, saturation = colorsys.rgb_to_hls(*to_rgb(color))
    return [
        colorsys.hls_to_rgb(
            (hue + hue_offset) % 1,
            np.clip(lightness + lightness_offset, 0.18, 0.78),
            saturation,
        )
        for hue_offset, lightness_offset in zip(
            np.linspace(-0.02, 0.02, count),
            np.linspace(-0.07, 0.17, count),
        )
    ]


def incidence(onsets, horizon=300, runs=40):
    times = np.sort(np.asarray(onsets, dtype=float))
    x = np.concatenate(([0], times, [horizon]))
    y = np.concatenate(([0], np.arange(1, len(times) + 1) / runs, [len(times) / runs]))
    return x, y


def mechanism(results):
    models = (
        ("Moshi", MOSHI, MOSHI_TRACE, results[0]),
        ("PersonaPlex", PERSONAPLEX, PERSONAPLEX_TRACE, results[1]),
    )
    fig, axes = paper_axes(100, ((22, 14, 93, 75), (144, 14, 215, 75)))
    grid = np.linspace(0, 300, 1201)
    prepared = []

    for name, color, trace_color, result in models:
        runs = result["baseline"]
        rate = result["frame_rate"]
        onsets = [
            run["onset_frame"] / rate for run in runs if run["onset_frame"] is not None
        ]
        observed = np.asarray([len(run["onset_probabilities"]) for run in runs])
        events = np.asarray([run["onset_frame"] is not None for run in runs])
        h0 = events.sum() / observed.sum()
        x, y = incidence(onsets, result["session_s"], len(runs))
        geometric = 1 - (1 - h0) ** (rate * grid)
        trajectories = [np.asarray(run["onset_probabilities"]) for run in runs]
        probability_limits = []
        for trajectory in trajectories:
            positive = trajectory[trajectory > 0]
            if positive.size:
                probability_limits.extend((positive.min(), positive.max()))
        # A fully censored, all-zero fixture still has a valid log-axis display.
        if not probability_limits:
            probability_limits = [1e-12, 1.0]
        prepared.append(
            (
                name,
                color,
                trace_color,
                rate,
                x,
                y,
                geometric,
                trajectories,
                max(0.1, np.ceil(10 * max(y.max(), geometric.max())) / 10),
                10 ** np.floor(np.log10(min(probability_limits))),
                1.5 * max(probability_limits),
            )
        )

    for ax, data in zip(axes, prepared):
        (
            name,
            color,
            trace_color,
            rate,
            x,
            y,
            geometric,
            trajectories,
            incidence_max,
            probability_min,
            probability_max,
        ) = data
        probability_ax = ax.twinx()
        probability_ax.set_zorder(0)
        ax.set_zorder(1)
        ax.patch.set_visible(False)

        ax.step(x, y, where="post", color=color, linewidth=1.6, zorder=4)
        ax.plot(grid, geometric, color=color, linewidth=1.2, linestyle=":", zorder=3)
        for trajectory, trajectory_color in zip(
            trajectories, shades(trace_color, len(trajectories))
        ):
            positive = trajectory[trajectory > 0]
            floor = positive.min() if positive.size else probability_min
            probability_ax.plot(
                np.arange(len(trajectory)) / rate,
                np.maximum(trajectory, floor),
                color=trajectory_color,
                linewidth=0.18,
                alpha=0.55,
                zorder=1,
            )

        ax.set_xlim(0, 300)
        ax.set_ylim(0, incidence_max)
        ax.set_yticks(np.arange(0, incidence_max + 0.01, 0.1))
        ax.set_xticks([0, 100, 200, 300])
        model_title(fig, ax, name, 100)
        style(ax, grid_axis=None)

        probability_ax.set_yscale("log")
        probability_ax.set_ylim(probability_min, probability_max)
        probability_ax.tick_params(
            colors=TEXT,
            labelsize=FONT_SIZE,
            width=0.65,
            length=2,
            pad=2,
        )
        probability_ax.minorticks_off()
        probability_ax.spines["right"].set_color(TEXT)
        probability_ax.spines["right"].set_linewidth(0.65)
        probability_ax.patch.set_visible(False)
        for tick in ax.get_xticks():
            probability_ax.axvline(tick, color=GRID, linewidth=0.5, zorder=0)
        for tick in ax.get_yticks():
            probability_ax.plot(
                [0, 1],
                [tick / incidence_max] * 2,
                color=GRID,
                linewidth=0.5,
                transform=probability_ax.transAxes,
                zorder=0,
            )

    footer = HPacker(
        children=[
            TextArea(
                "Onset time (s);",
                textprops={
                    "fontsize": FONT_SIZE,
                    "fontweight": "bold",
                    "color": TEXT,
                },
            ),
            TextArea(
                " left: incidence; right: probability",
                textprops={
                    "fontsize": FONT_SIZE,
                    "color": TEXT,
                },
            ),
        ],
        align="baseline",
        pad=0,
        sep=0,
    )
    fig.add_artist(
        AnchoredOffsetbox(
            loc="lower center",
            child=footer,
            frameon=False,
            pad=0,
            borderpad=0,
            bbox_to_anchor=(0.5, 0.005),
            bbox_transform=fig.transFigure,
        )
    )
    fig.savefig(OUT / "mechanism.pdf")
    plt.close(fig)


def counterfactual(results):
    models = []
    for name, color, result in zip(
        ("Moshi", "PersonaPlex"), (MOSHI, PERSONAPLEX), results
    ):
        idle = np.asarray(
            [
                run["scores"][0]["score"]
                for run in result["calibration_idle"]
                if run["scores"]
            ]
        )
        speech = np.asarray(
            [
                run["score"]
                for run in result["calibration_responses"]
                if run["score"] is not None
            ]
        )
        boundaries = result["boundaries"]
        thresholds = (
            boundaries["response"],
            boundaries["balanced"],
            boundaries["suppression"],
        )
        models.append((name, color, (idle, speech), thresholds, thresholds[::2]))

    styles = ((0, (1, 1.5)), "-", (0, (5, 2)))
    labels = ("preserve", "balanced", "suppress")
    line_order = (1, 0, 2)
    fig, axes = paper_axes(96, ((36, 14, 133, 59), (148, 14, 242, 59)))
    positions = (0, 0.72)

    for model_index, (ax, (name, color, scores, thresholds, balanced)) in enumerate(
        zip(axes, models)
    ):
        for score_index, (y, values) in enumerate(zip(positions, scores)):
            if len(values) > 2 and np.ptp(values) > 0:
                violin = ax.violinplot(
                    values,
                    positions=[y],
                    vert=False,
                    widths=0.5,
                    showextrema=False,
                )["bodies"][0]
                violin.set_facecolor(color)
                violin.set_edgecolor(color)
                violin.set_alpha(0.22)
                vertices = violin.get_paths()[0].vertices
                vertices[:, 1] = np.maximum(vertices[:, 1], y)
            jitter = np.random.default_rng(10 * model_index + score_index).uniform(
                -0.14, -0.05, len(values)
            )
            ax.scatter(
                values,
                y + jitter,
                color=color,
                s=4,
                alpha=0.85,
                edgecolors="none",
            )
        ax.axvspan(*balanced, color="#b5b5b5", alpha=0.28, linewidth=0)
        for index in line_order:
            ax.axvline(
                thresholds[index],
                color="#b5b5b5" if index == 1 else TEXT,
                linestyle=styles[index],
                linewidth=1.4 if index == 1 else 0.8,
                zorder=4,
            )
        model_title(fig, ax, name, 96)
        upper = (
            np.ceil(
                10
                * max(
                    *(values.max() for values in scores if values.size),
                    *thresholds,
                    0.1,
                )
            )
            / 10
        )
        ax.set_xlim(-0.015, upper)
        ax.set_ylim(-0.24, 1.02)
        ax.set_yticks(positions, ("spurious", "response"))
        ax.set_xticks(np.arange(0, upper + 0.001, 0.2))
        style(ax, grid_axis="x")
        ax.tick_params(axis="y", length=0)
        if model_index:
            ax.tick_params(axis="y", labelleft=False)

    fig.text(
        0.5,
        1 - 81 / 96,
        "Divergence (Dₜ)",
        ha="center",
        va="baseline",
        fontsize=FONT_SIZE,
        fontweight="bold",
        color=TEXT,
    )
    fig.legend(
        handles=[
            Line2D(
                [],
                [],
                color="#b5b5b5" if index == 1 else TEXT,
                linestyle=styles[index],
                linewidth=1.0,
                label=label,
            )
            for index, label in enumerate(labels)
        ],
        frameon=False,
        fontsize=FONT_SIZE,
        labelcolor=TEXT,
        handlelength=1.6,
        loc="lower left",
        bbox_to_anchor=(0, 0.5 / 96, 216 / WIDTH, 1),
        bbox_transform=fig.transFigure,
        mode="expand",
        ncol=3,
        borderaxespad=0,
        borderpad=0,
    )
    fig.savefig(OUT / "counterfactual.pdf")
    plt.close(fig)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("moshi", type=Path)
    parser.add_argument("personaplex", type=Path)
    args = parser.parse_args()
    results = tuple(
        json.loads(path.read_text()) for path in (args.moshi, args.personaplex)
    )
    OUT.mkdir(exist_ok=True)
    mechanism(results)
    counterfactual(results)


if __name__ == "__main__":
    main()
