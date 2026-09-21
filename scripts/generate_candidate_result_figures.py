from __future__ import annotations

import argparse
import csv
from pathlib import Path


PALETTE = {
    "fisher": "#1f77b4",
    "fixed": "#ff7f0e",
    "random": "#2ca02c",
    "smallest": "#9467bd",
    "identity_cost": "#8c564b",
}

DISPLAY_NAMES = {
    "fisher": "Fisher",
    "fixed": "Fixed",
    "random": "Random",
    "smallest": "Smallest",
    "identity_cost": "Identity cost",
    "center_mean": "Center mean",
    "copy_move": "Copy-move",
    "constant_average_block": "Constant avg.",
    "inter_block_substitution": "Inter-block subst.",
}


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def esc(text: object) -> str:
    return (
        str(text)
        .replace("&", "&amp;")
        .replace("<", "&lt;")
        .replace(">", "&gt;")
        .replace('"', "&quot;")
    )


def text(x: float, y: float, value: object, size: int = 12, anchor: str = "middle") -> str:
    return (
        f'<text x="{x:.1f}" y="{y:.1f}" font-family="Arial, sans-serif" '
        f'font-size="{size}" text-anchor="{anchor}" fill="#222">{esc(value)}</text>'
    )


def rect(x: float, y: float, width: float, height: float, fill: str) -> str:
    return (
        f'<rect x="{x:.1f}" y="{y:.1f}" width="{width:.1f}" '
        f'height="{height:.1f}" fill="{fill}"/>'
    )


def write_svg(path: Path, width: int, height: int, body: list[str]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    content = [
        f'<svg xmlns="http://www.w3.org/2000/svg" width="{width}" height="{height}" '
        f'viewBox="0 0 {width} {height}" role="img">',
        '<rect width="100%" height="100%" fill="white"/>',
        *body,
        "</svg>",
        "",
    ]
    path.write_text("\n".join(content), encoding="utf-8")


def attack_f1_figure(rows: list[dict[str, str]], output: Path) -> None:
    attacks = [
        "center_mean",
        "copy_move",
        "constant_average_block",
        "inter_block_substitution",
    ]
    modes = ["fisher", "fixed", "random", "identity_cost", "smallest"]
    by_mode = {row["basisMode"]: row for row in rows}
    width, height = 980, 520
    left, right, top, bottom = 88, 30, 70, 120
    plot_w = width - left - right
    plot_h = height - top - bottom
    y_min, y_max = 0.86, 1.0

    body = [
        text(width / 2, 32, "Candidate v2: block-level F1 by attack", 18),
        text(width / 2, 54, "Restricted DTD+COCO pilot, 20 evaluation images", 12),
    ]
    for tick in [0.86, 0.90, 0.94, 0.98, 1.00]:
        y = top + (y_max - tick) / (y_max - y_min) * plot_h
        body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
        body.append(text(left - 12, y + 4, f"{tick:.2f}", 11, "end"))
    body.append(f'<line x1="{left}" y1="{top}" x2="{left}" y2="{top+plot_h}" stroke="#444"/>')
    body.append(f'<line x1="{left}" y1="{top+plot_h}" x2="{width-right}" y2="{top+plot_h}" stroke="#444"/>')

    group_w = plot_w / len(attacks)
    bar_gap = 4
    bar_w = (group_w - 34) / len(modes)
    for attack_index, attack in enumerate(attacks):
        group_x = left + attack_index * group_w + 17
        for mode_index, mode in enumerate(modes):
            value = float(by_mode[mode][attack])
            bar_h = (value - y_min) / (y_max - y_min) * plot_h
            x = group_x + mode_index * bar_w
            y = top + plot_h - bar_h
            body.append(rect(x, y, bar_w - bar_gap, bar_h, PALETTE[mode]))
            if mode == "fisher":
                body.append(text(x + (bar_w - bar_gap) / 2, y - 5, f"{value:.3f}", 10))
        body.append(text(group_x + (len(modes) * bar_w) / 2 - 8, top + plot_h + 34, DISPLAY_NAMES[attack], 11))

    legend_x, legend_y = left, height - 52
    for index, mode in enumerate(modes):
        x = legend_x + index * 165
        body.append(rect(x, legend_y, 16, 16, PALETTE[mode]))
        body.append(text(x + 24, legend_y + 13, DISPLAY_NAMES[mode], 12, "start"))
    body.append(text(24, top + plot_h / 2, "F1", 12, "middle"))
    write_svg(output, width, height, body)


def clean_metrics_figure(rows: list[dict[str, str]], output: Path) -> None:
    modes = ["fisher", "fixed", "random", "smallest", "identity_cost"]
    by_mode = {row["basisMode"]: row for row in rows}
    width, height = 980, 520
    left, right, top = 84, 34, 72
    panel_gap = 70
    panel_h = 145
    panel_w = width - left - right
    psnr_min, psnr_max = 54.0, 62.0
    flags_min, flags_max = 0.0, 12.0

    body = [
        text(width / 2, 32, "Candidate v2: clean-image behavior", 18),
        text(width / 2, 54, "Mean clean PSNR and clean flagged blocks", 12),
    ]

    def panel(y0: float, y_min: float, y_max: float, key: str, label: str, value_fmt: str) -> None:
        for tick in [y_min, (y_min + y_max) / 2, y_max]:
            y = y0 + (y_max - tick) / (y_max - y_min) * panel_h
            body.append(f'<line x1="{left}" y1="{y:.1f}" x2="{width-right}" y2="{y:.1f}" stroke="#e6e6e6"/>')
            body.append(text(left - 12, y + 4, f"{tick:.1f}", 11, "end"))
        body.append(f'<line x1="{left}" y1="{y0}" x2="{left}" y2="{y0+panel_h}" stroke="#444"/>')
        body.append(f'<line x1="{left}" y1="{y0+panel_h}" x2="{width-right}" y2="{y0+panel_h}" stroke="#444"/>')
        bar_w = panel_w / len(modes) * 0.62
        for index, mode in enumerate(modes):
            value = float(by_mode[mode][key])
            bar_h = (value - y_min) / (y_max - y_min) * panel_h
            x = left + index * (panel_w / len(modes)) + 34
            y = y0 + panel_h - bar_h
            body.append(rect(x, y, bar_w, bar_h, PALETTE[mode]))
            body.append(text(x + bar_w / 2, y - 6, value_fmt.format(value), 10))
            if key == "cleanFlaggedBlockCount":
                body.append(text(x + bar_w / 2, y0 + panel_h + 28, DISPLAY_NAMES[mode], 11))
        body.append(text(24, y0 + panel_h / 2, label, 12, "middle"))

    panel(top, psnr_min, psnr_max, "meanCleanPsnrDb", "PSNR (dB)", "{:.2f}")
    panel(top + panel_h + panel_gap, flags_min, flags_max, "cleanFlaggedBlockCount", "Clean flags", "{:.0f}")
    write_svg(output, width, height, body)


def generate_figures(input_dir: Path, output_dir: Path) -> dict[str, str]:
    attack_rows = read_csv(input_dir / "attack_f1.csv")
    clean_rows = read_csv(input_dir / "clean_metrics.csv")
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "attack_f1": output_dir / "attack_f1_by_mode.svg",
        "clean_metrics": output_dir / "clean_metrics_by_mode.svg",
    }
    attack_f1_figure(attack_rows, paths["attack_f1"])
    clean_metrics_figure(clean_rows, paths["clean_metrics"])
    return {name: path.as_posix() for name, path in paths.items()}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate SVG figures from candidate result CSV tables."
    )
    parser.add_argument("--input-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    return parser.parse_args()


def main() -> None:
    paths = generate_figures(**vars(parse_args()))
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
