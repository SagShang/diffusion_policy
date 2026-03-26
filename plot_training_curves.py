#!/usr/bin/env python3
import argparse
import json
import math
from html import escape
from pathlib import Path
from typing import Dict, List, Sequence, Tuple


PANEL_METRICS: Sequence[Tuple[str, str]] = (
    ("train_loss", "Train Loss"),
    ("val_loss", "Validation Loss"),
    ("train_action_mse_error", "Train Action MSE"),
    ("lr", "Learning Rate"),
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Plot training curves from logs.json.txt into an SVG file."
    )
    parser.add_argument(
        "experiment_dirs",
        nargs="+",
        type=Path,
        help="Experiment output directories that contain logs.json.txt.",
    )
    parser.add_argument(
        "--log-name",
        default="logs.json.txt",
        help="Name of the JSON-lines log file inside each experiment directory.",
    )
    parser.add_argument(
        "--output-name",
        default="training_curves.svg",
        help="Output SVG filename written into each experiment directory.",
    )
    parser.add_argument(
        "--ema-alpha",
        type=float,
        default=0.03,
        help="EMA smoothing factor used for train_loss.",
    )
    parser.add_argument(
        "--max-points",
        type=int,
        default=2500,
        help="Maximum points kept per polyline after downsampling.",
    )
    return parser.parse_args()


def is_number(value: object) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool)


def parse_log(log_path: Path) -> Dict[str, Dict[str, List[float]]]:
    series: Dict[str, Dict[str, List[float]]] = {}
    with log_path.open("r", encoding="utf-8") as f:
        for line_idx, line in enumerate(f):
            if not line.endswith("\n"):
                break
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            global_step = float(record.get("global_step", line_idx))
            epoch = float(record.get("epoch", line_idx))
            for key, value in record.items():
                if key in ("global_step", "epoch"):
                    continue
                if not is_number(value):
                    continue
                data = series.setdefault(key, {"x": [], "y": [], "epoch": []})
                data["x"].append(global_step)
                data["y"].append(float(value))
                data["epoch"].append(epoch)
    return series


def downsample(
    xs: Sequence[float], ys: Sequence[float], max_points: int
) -> Tuple[List[float], List[float]]:
    if len(xs) <= max_points:
        return list(xs), list(ys)
    if max_points < 2:
        return [xs[0]], [ys[0]]
    last_index = len(xs) - 1
    step = last_index / (max_points - 1)
    indices = sorted({round(i * step) for i in range(max_points)})
    if indices[-1] != last_index:
        indices.append(last_index)
    return [xs[i] for i in indices], [ys[i] for i in indices]


def ema(values: Sequence[float], alpha: float) -> List[float]:
    if not values:
        return []
    smoothed = [values[0]]
    running = values[0]
    for value in values[1:]:
        running = alpha * value + (1.0 - alpha) * running
        smoothed.append(running)
    return smoothed


def nice_number(value: float, round_value: bool) -> float:
    if value <= 0:
        return 1.0
    exponent = math.floor(math.log10(value))
    fraction = value / (10 ** exponent)
    if round_value:
        if fraction < 1.5:
            nice_fraction = 1.0
        elif fraction < 3.0:
            nice_fraction = 2.0
        elif fraction < 7.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    else:
        if fraction <= 1.0:
            nice_fraction = 1.0
        elif fraction <= 2.0:
            nice_fraction = 2.0
        elif fraction <= 5.0:
            nice_fraction = 5.0
        else:
            nice_fraction = 10.0
    return nice_fraction * (10 ** exponent)


def make_ticks(vmin: float, vmax: float, tick_count: int = 5) -> List[float]:
    if not math.isfinite(vmin) or not math.isfinite(vmax):
        return [0.0, 1.0]
    if vmin == vmax:
        padding = abs(vmin) * 0.05 or 1.0
        vmin -= padding
        vmax += padding
    span = vmax - vmin
    step = nice_number(span / max(tick_count - 1, 1), round_value=True)
    nice_min = math.floor(vmin / step) * step
    nice_max = math.ceil(vmax / step) * step
    ticks = []
    value = nice_min
    max_iterations = 100
    iterations = 0
    while value <= nice_max + step * 0.5 and iterations < max_iterations:
        ticks.append(0.0 if abs(value) < step * 1e-9 else value)
        value += step
        iterations += 1
    return ticks


def format_number(value: float) -> str:
    absolute = abs(value)
    if absolute == 0:
        return "0"
    if absolute >= 1000 or absolute < 1e-3:
        return f"{value:.2e}"
    if absolute >= 1:
        return f"{value:.4f}".rstrip("0").rstrip(".")
    return f"{value:.5f}".rstrip("0").rstrip(".")


def svg_polyline(
    xs: Sequence[float],
    ys: Sequence[float],
    color: str,
    stroke_width: float,
    opacity: float = 1.0,
) -> str:
    points = " ".join(f"{x:.2f},{y:.2f}" for x, y in zip(xs, ys))
    return (
        f'<polyline fill="none" stroke="{color}" stroke-width="{stroke_width}" '
        f'stroke-opacity="{opacity}" stroke-linecap="round" stroke-linejoin="round" '
        f'points="{points}" />'
    )


def map_to_pixels(
    xs: Sequence[float],
    ys: Sequence[float],
    x_min: float,
    x_max: float,
    y_min: float,
    y_max: float,
    left: float,
    top: float,
    width: float,
    height: float,
) -> Tuple[List[float], List[float]]:
    x_span = x_max - x_min or 1.0
    y_span = y_max - y_min or 1.0
    px = [left + ((x - x_min) / x_span) * width for x in xs]
    py = [top + height - ((y - y_min) / y_span) * height for y in ys]
    return px, py


def latest_value(series: Dict[str, List[float]]) -> float:
    return series["y"][-1]


def build_panel(
    metric_key: str,
    metric_title: str,
    series: Dict[str, Dict[str, List[float]]],
    left: float,
    top: float,
    width: float,
    height: float,
    x_limit: float,
    max_points: int,
    ema_alpha: float,
) -> str:
    parts: List[str] = []
    panel_fill = "#fffdf8"
    border = "#cfc8ba"
    grid = "#e6dfd1"
    text = "#2f2a25"
    muted = "#6d6458"

    parts.append(
        f'<rect x="{left}" y="{top}" width="{width}" height="{height}" rx="18" '
        f'fill="{panel_fill}" stroke="{border}" stroke-width="1.2" />'
    )

    header_x = left + 22
    header_y = top + 28
    parts.append(
        f'<text x="{header_x}" y="{header_y}" font-size="20" font-weight="700" fill="{text}" '
        f'font-family="Arial, Helvetica, sans-serif">{escape(metric_title)}</text>'
    )

    if metric_key not in series or not series[metric_key]["x"]:
        parts.append(
            f'<text x="{header_x}" y="{header_y + 28}" font-size="14" fill="{muted}" '
            f'font-family="Arial, Helvetica, sans-serif">No data</text>'
        )
        return "\n".join(parts)

    data = series[metric_key]
    xs = data["x"]
    ys = data["y"]
    x_min = 0.0
    x_max = max(x_limit, xs[-1] if xs else 1.0)
    y_min = min(ys)
    y_max = max(ys)
    if y_min == y_max:
        pad = abs(y_min) * 0.05 or 1.0
        y_min -= pad
        y_max += pad
    else:
        pad = (y_max - y_min) * 0.08
        y_min -= pad
        y_max += pad

    plot_left = left + 66
    plot_top = top + 48
    plot_width = width - 88
    plot_height = height - 96

    x_ticks = make_ticks(x_min, x_max, tick_count=6)
    y_ticks = make_ticks(y_min, y_max, tick_count=5)

    for x_tick in x_ticks:
        x_px, _ = map_to_pixels(
            [x_tick],
            [y_min],
            x_min,
            x_max,
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        parts.append(
            f'<line x1="{x_px[0]:.2f}" y1="{plot_top:.2f}" x2="{x_px[0]:.2f}" y2="{plot_top + plot_height:.2f}" '
            f'stroke="{grid}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{x_px[0]:.2f}" y="{plot_top + plot_height + 28:.2f}" text-anchor="middle" '
            f'font-size="12" fill="{muted}" font-family="Arial, Helvetica, sans-serif">{escape(format_number(x_tick))}</text>'
        )

    for y_tick in y_ticks:
        _, y_px = map_to_pixels(
            [x_min],
            [y_tick],
            x_min,
            x_max,
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        parts.append(
            f'<line x1="{plot_left:.2f}" y1="{y_px[0]:.2f}" x2="{plot_left + plot_width:.2f}" y2="{y_px[0]:.2f}" '
            f'stroke="{grid}" stroke-width="1" />'
        )
        parts.append(
            f'<text x="{plot_left - 10:.2f}" y="{y_px[0] + 4:.2f}" text-anchor="end" '
            f'font-size="12" fill="{muted}" font-family="Arial, Helvetica, sans-serif">{escape(format_number(y_tick))}</text>'
        )

    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top + plot_height:.2f}" x2="{plot_left + plot_width:.2f}" y2="{plot_top + plot_height:.2f}" '
        f'stroke="{border}" stroke-width="1.4" />'
    )
    parts.append(
        f'<line x1="{plot_left:.2f}" y1="{plot_top:.2f}" x2="{plot_left:.2f}" y2="{plot_top + plot_height:.2f}" '
        f'stroke="{border}" stroke-width="1.4" />'
    )

    if metric_key == "train_loss":
        raw_x, raw_y = downsample(xs, ys, max_points=max_points)
        smooth_y = ema(ys, alpha=ema_alpha)
        smooth_x, smooth_y = downsample(xs, smooth_y, max_points=max_points)
        raw_px, raw_py = map_to_pixels(
            raw_x,
            raw_y,
            x_min,
            x_max,
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        smooth_px, smooth_py = map_to_pixels(
            smooth_x,
            smooth_y,
            x_min,
            x_max,
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        parts.append(svg_polyline(raw_px, raw_py, color="#8ea7b8", stroke_width=1.4, opacity=0.45))
        parts.append(svg_polyline(smooth_px, smooth_py, color="#0f4c5c", stroke_width=2.4))
        legend_y = top + height - 18
        parts.append(
            f'<line x1="{header_x}" y1="{legend_y - 4}" x2="{header_x + 18}" y2="{legend_y - 4}" '
            f'stroke="#8ea7b8" stroke-width="2" stroke-opacity="0.6" />'
        )
        parts.append(
            f'<text x="{header_x + 24}" y="{legend_y}" font-size="12" fill="{muted}" '
            f'font-family="Arial, Helvetica, sans-serif">raw</text>'
        )
        parts.append(
            f'<line x1="{header_x + 64}" y1="{legend_y - 4}" x2="{header_x + 82}" y2="{legend_y - 4}" '
            f'stroke="#0f4c5c" stroke-width="2.4" />'
        )
        parts.append(
            f'<text x="{header_x + 88}" y="{legend_y}" font-size="12" fill="{muted}" '
            f'font-family="Arial, Helvetica, sans-serif">EMA</text>'
        )
    else:
        color_map = {
            "val_loss": "#bc4b51",
            "train_action_mse_error": "#386641",
            "lr": "#8a5a44",
        }
        line_x, line_y = downsample(xs, ys, max_points=max_points)
        line_px, line_py = map_to_pixels(
            line_x,
            line_y,
            x_min,
            x_max,
            y_min,
            y_max,
            plot_left,
            plot_top,
            plot_width,
            plot_height,
        )
        parts.append(
            svg_polyline(
                line_px,
                line_py,
                color=color_map.get(metric_key, "#0f4c5c"),
                stroke_width=2.4,
            )
        )

    latest = latest_value(data)
    parts.append(
        f'<text x="{left + width - 22}" y="{header_y}" text-anchor="end" font-size="13" fill="{muted}" '
        f'font-family="Arial, Helvetica, sans-serif">latest: {escape(format_number(latest))}</text>'
    )
    parts.append(
        f'<text x="{plot_left + plot_width}" y="{plot_top + plot_height + 28:.2f}" text-anchor="end" '
        f'font-size="12" fill="{muted}" font-family="Arial, Helvetica, sans-serif">global_step</text>'
    )
    return "\n".join(parts)


def build_svg(
    experiment_dir: Path,
    series: Dict[str, Dict[str, List[float]]],
    output_name: str,
    max_points: int,
    ema_alpha: float,
) -> str:
    width = 1440
    height = 1080
    background = "#f4efe6"
    text = "#2f2a25"
    muted = "#6d6458"

    max_step = 1.0
    for data in series.values():
        if data["x"]:
            max_step = max(max_step, data["x"][-1])

    summary_bits = []
    for metric_key, title in PANEL_METRICS:
        if metric_key in series and series[metric_key]["y"]:
            summary_bits.append(f"{title}: {format_number(series[metric_key]['y'][-1])}")
    summary_text = " | ".join(summary_bits)

    parts: List[str] = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" viewBox="0 0 {width} {height}">',
        f'<rect width="{width}" height="{height}" fill="{background}" />',
        (
            '<text x="48" y="64" font-size="30" font-weight="700" fill="{text}" '
            'font-family="Arial, Helvetica, sans-serif">{title}</text>'
        ).format(text=text, title=escape(experiment_dir.name)),
        (
            '<text x="48" y="92" font-size="15" fill="{muted}" '
            'font-family="Arial, Helvetica, sans-serif">{subtitle}</text>'
        ).format(
            muted=muted,
            subtitle=escape(f"log: {experiment_dir / 'logs.json.txt'}"),
        ),
        (
            '<text x="48" y="118" font-size="14" fill="{muted}" '
            'font-family="Arial, Helvetica, sans-serif">{summary}</text>'
        ).format(muted=muted, summary=escape(summary_text)),
    ]

    panel_width = 652
    panel_height = 390
    left_positions = [48, 740]
    top_positions = [160, 582]
    panel_specs = [
        (left_positions[0], top_positions[0], PANEL_METRICS[0]),
        (left_positions[1], top_positions[0], PANEL_METRICS[1]),
        (left_positions[0], top_positions[1], PANEL_METRICS[2]),
        (left_positions[1], top_positions[1], PANEL_METRICS[3]),
    ]

    for left, top, (metric_key, metric_title) in panel_specs:
        parts.append(
            build_panel(
                metric_key=metric_key,
                metric_title=metric_title,
                series=series,
                left=left,
                top=top,
                width=panel_width,
                height=panel_height,
                x_limit=max_step,
                max_points=max_points,
                ema_alpha=ema_alpha,
            )
        )

    parts.append(
        (
            '<text x="{x}" y="{y}" text-anchor="end" font-size="13" fill="{muted}" '
            'font-family="Arial, Helvetica, sans-serif">saved as {name}</text>'
        ).format(x=width - 48, y=height - 28, muted=muted, name=escape(output_name))
    )
    parts.append("</svg>")
    return "\n".join(parts)


def plot_experiment(
    experiment_dir: Path,
    log_name: str,
    output_name: str,
    ema_alpha: float,
    max_points: int,
) -> Path:
    log_path = experiment_dir / log_name
    if not log_path.is_file():
        raise FileNotFoundError(f"Missing log file: {log_path}")
    series = parse_log(log_path)
    svg = build_svg(
        experiment_dir=experiment_dir,
        series=series,
        output_name=output_name,
        max_points=max_points,
        ema_alpha=ema_alpha,
    )
    output_path = experiment_dir / output_name
    output_path.write_text(svg, encoding="utf-8")
    return output_path


def main() -> None:
    args = parse_args()
    for experiment_dir in args.experiment_dirs:
        output_path = plot_experiment(
            experiment_dir=experiment_dir,
            log_name=args.log_name,
            output_name=args.output_name,
            ema_alpha=args.ema_alpha,
            max_points=args.max_points,
        )
        print(output_path)


if __name__ == "__main__":
    main()
