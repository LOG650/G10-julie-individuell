from __future__ import annotations

import os
import re
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import Any

os.environ.setdefault(
    "MPLCONFIGDIR",
    str(Path(tempfile.gettempdir()) / "g10-julie-individuell-matplotlib"),
)

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt


ROOT = Path(__file__).resolve().parents[2]
DATA_PATH = ROOT / "004 data" / "Data som skal brukes Anonymisert.csv"
VISUALIZATION_DIR = Path(__file__).resolve().parent
FIGURES_DIR = VISUALIZATION_DIR / "figures"
TABLES_DIR = VISUALIZATION_DIR / "tables"

MONTH_ORDER = {
    "Januar": 1,
    "Februar": 2,
    "Mars": 3,
    "April": 4,
    "Mai": 5,
    "Juni": 6,
    "Juli": 7,
    "August": 8,
    "September": 9,
    "Oktober": 10,
    "November": 11,
    "Desember": 12,
}
MONTH_COLUMNS = list(MONTH_ORDER.keys())
YEAR_PATTERN = re.compile(r"År:\s*(\d{4})")

PRIMARY_COLOR = "#0F4C5C"
SECONDARY_COLOR = "#2C7A7B"
ACCENT_COLOR = "#D97706"
GRID_COLOR = "#D9E2EC"
TEXT_COLOR = "#102A43"
BAR_COLORS = ["#0F4C5C", "#136F63", "#1F9D8B", "#5FB49C", "#C8D5B9"]
HEATMAP_CMAP = "YlOrRd"


@dataclass
class DatasetStats:
    raw_rows: int
    year_rows: int
    header_rows: int
    vessel_rows: int


def ensure_output_dirs() -> None:
    FIGURES_DIR.mkdir(parents=True, exist_ok=True)
    TABLES_DIR.mkdir(parents=True, exist_ok=True)


def vessel_sort_key(vessel_name: str) -> tuple[int, str]:
    match = re.search(r"(\d+)$", vessel_name)
    if match:
        return int(match.group(1)), vessel_name
    return 10_000, vessel_name


def save_figure(fig: plt.Figure, filename: str) -> None:
    fig.savefig(FIGURES_DIR / filename, dpi=220, bbox_inches="tight")
    plt.close(fig)


def format_markdown_value(value: Any) -> str:
    if pd.isna(value):
        return ""
    if isinstance(value, (float, np.floating)):
        return f"{float(value):.2f}"
    return str(value).replace("|", "\\|")


def dataframe_to_markdown(df: pd.DataFrame) -> str:
    if df.empty:
        return "_Ingen rader tilgjengelig._"

    headers = df.columns.tolist()
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for _, row in df.iterrows():
        values = [format_markdown_value(row[column]) for column in headers]
        lines.append("| " + " | ".join(values) + " |")
    return "\n".join(lines)


def write_markdown_table(
    filename: str,
    title: str,
    sections: list[tuple[str, pd.DataFrame]],
) -> None:
    lines = [f"# {title}", ""]
    for index, (heading, frame) in enumerate(sections):
        if index > 0:
            lines.append("")
        lines.extend([f"## {heading}", "", dataframe_to_markdown(frame)])
    (TABLES_DIR / filename).write_text("\n".join(lines) + "\n", encoding="utf-8")


def save_table(df: pd.DataFrame, basename: str, title: str) -> None:
    df.to_csv(TABLES_DIR / f"{basename}.csv", index=False, encoding="utf-8-sig")
    write_markdown_table(f"{basename}.md", title, [("Tabell", df)])


def configure_matplotlib() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "axes.edgecolor": "#BCCCDC",
            "axes.labelcolor": TEXT_COLOR,
            "axes.titlecolor": TEXT_COLOR,
            "axes.titlesize": 14,
            "axes.labelsize": 11,
            "xtick.color": TEXT_COLOR,
            "ytick.color": TEXT_COLOR,
            "grid.color": GRID_COLOR,
            "grid.linestyle": "--",
            "grid.linewidth": 0.7,
            "font.size": 10,
            "legend.frameon": False,
        }
    )


def load_historical_dataset() -> tuple[pd.DataFrame, DatasetStats]:
    raw_df = pd.read_csv(DATA_PATH, sep=";", header=None, encoding="utf-8-sig", dtype=str)

    records: list[dict[str, Any]] = []
    current_year: int | None = None
    year_rows = 0
    header_rows = 0
    vessel_rows = 0

    for row in raw_df.itertuples(index=False, name=None):
        values = ["" if pd.isna(value) else str(value).strip() for value in row]
        if not any(values):
            continue

        label = values[1] if len(values) > 1 else ""
        year_match = YEAR_PATTERN.fullmatch(label)
        if year_match:
            current_year = int(year_match.group(1))
            year_rows += 1
            continue

        if label == "Måned/Fartøy":
            header_rows += 1
            continue

        if not label.startswith("Fartøy ") or current_year is None:
            continue

        vessel_rows += 1
        special_requirements = values[14] if len(values) > 14 else ""

        for column_index, month_name in enumerate(MONTH_COLUMNS, start=2):
            raw_value = values[column_index] if len(values) > column_index else ""
            if raw_value in {"", "N/A"}:
                continue

            numeric_value = raw_value.replace("%", "").replace(" ", "").replace(",", ".")
            if numeric_value == "":
                continue

            records.append(
                {
                    "vessel": re.sub(r"\s+", " ", label).strip(),
                    "month_name": month_name,
                    "month_num": MONTH_ORDER[month_name],
                    "date": pd.Timestamp(
                        year=current_year,
                        month=MONTH_ORDER[month_name],
                        day=1,
                    ),
                    "offhire_pct": float(numeric_value),
                    "special_requirements": special_requirements,
                }
            )

    if not records:
        raise RuntimeError("Fant ingen gyldige observasjoner i datasettet.")

    panel_df = pd.DataFrame.from_records(records)
    panel_df = panel_df.sort_values(["vessel", "date"]).reset_index(drop=True)
    panel_df["year"] = panel_df["date"].dt.year
    panel_df["month_label"] = panel_df["date"].dt.strftime("%Y-%m")

    stats = DatasetStats(
        raw_rows=len(raw_df),
        year_rows=year_rows,
        header_rows=header_rows,
        vessel_rows=vessel_rows,
    )
    return panel_df, stats


def build_data_coverage_summary(panel_df: pd.DataFrame, stats: DatasetStats) -> pd.DataFrame:
    min_date = panel_df["date"].min()
    max_date = panel_df["date"].max()
    unique_months = panel_df["date"].nunique()
    summary_rows = [
        {"Metrikk": "Tidsperiode", "Verdi": f"{min_date:%Y-%m} til {max_date:%Y-%m}"},
        {"Metrikk": "Antall kalendermåneder", "Verdi": unique_months},
        {"Metrikk": "Antall fartøy", "Verdi": panel_df["vessel"].nunique()},
        {"Metrikk": "Antall rå rader i CSV", "Verdi": stats.raw_rows},
        {"Metrikk": "Antall årsblokker", "Verdi": stats.year_rows},
        {"Metrikk": "Antall header-rader", "Verdi": stats.header_rows},
        {"Metrikk": "Antall fartøyrader", "Verdi": stats.vessel_rows},
        {"Metrikk": "Antall rensede observasjoner", "Verdi": len(panel_df)},
        {
            "Metrikk": "Ufullstendige år",
            "Verdi": "2021 starter i april, 2026 slutter i mars",
        },
    ]
    return pd.DataFrame(summary_rows)


def build_observations_per_year(panel_df: pd.DataFrame) -> pd.DataFrame:
    year_counts = (
        panel_df.groupby("year")
        .size()
        .reset_index(name="Antall observasjoner")
        .rename(columns={"year": "År"})
    )
    return year_counts


def build_vessel_summary(panel_df: pd.DataFrame) -> pd.DataFrame:
    summary = (
        panel_df.groupby("vessel")
        .agg(
            Observasjoner=("offhire_pct", "size"),
            Gjennomsnitt_offhire_pct=("offhire_pct", "mean"),
            Median_offhire_pct=("offhire_pct", "median"),
            Maks_offhire_pct=("offhire_pct", "max"),
            Nullmåneder=("offhire_pct", lambda values: int((values == 0).sum())),
            Første_måned=("date", "min"),
            Siste_måned=("date", "max"),
        )
        .reset_index()
        .rename(columns={"vessel": "Fartøy"})
    )
    summary["Andel_nullmåneder_pct"] = (
        summary["Nullmåneder"] / summary["Observasjoner"] * 100
    )
    summary["Første_måned"] = pd.to_datetime(summary["Første_måned"]).dt.strftime("%Y-%m")
    summary["Siste_måned"] = pd.to_datetime(summary["Siste_måned"]).dt.strftime("%Y-%m")

    summary = summary[
        [
            "Fartøy",
            "Observasjoner",
            "Gjennomsnitt_offhire_pct",
            "Median_offhire_pct",
            "Maks_offhire_pct",
            "Andel_nullmåneder_pct",
            "Første_måned",
            "Siste_måned",
        ]
    ]
    summary = summary.sort_values(
        ["Gjennomsnitt_offhire_pct", "Fartøy"], ascending=[False, True]
    ).reset_index(drop=True)
    numeric_columns = [
        "Gjennomsnitt_offhire_pct",
        "Median_offhire_pct",
        "Maks_offhire_pct",
        "Andel_nullmåneder_pct",
    ]
    summary[numeric_columns] = summary[numeric_columns].round(2)
    return summary


def build_top_observations(panel_df: pd.DataFrame, limit: int = 10) -> pd.DataFrame:
    top_rows = (
        panel_df.sort_values(["offhire_pct", "date"], ascending=[False, True])
        .head(limit)
        .copy()
    )
    top_rows["Måned"] = top_rows["date"].dt.strftime("%Y-%m")
    result = top_rows[["vessel", "Måned", "offhire_pct"]].rename(
        columns={"vessel": "Fartøy", "offhire_pct": "Offhire_pct"}
    )
    result["Offhire_pct"] = result["Offhire_pct"].round(2)
    return result.reset_index(drop=True)


def plot_monthly_total_offhire(panel_df: pd.DataFrame) -> None:
    monthly = (
        panel_df.groupby("date")
        .agg(
            total_offhire_pct=("offhire_pct", "sum"),
            mean_offhire_pct=("offhire_pct", "mean"),
            observed_vessels=("vessel", "nunique"),
        )
        .reset_index()
    )
    max_row = monthly.loc[monthly["total_offhire_pct"].idxmax()]

    fig, ax = plt.subplots(figsize=(12, 5.8))
    ax.plot(
        monthly["date"],
        monthly["total_offhire_pct"],
        color=PRIMARY_COLOR,
        linewidth=2.4,
        label="Samlet offhire",
    )
    ax.fill_between(
        monthly["date"],
        monthly["total_offhire_pct"],
        color=SECONDARY_COLOR,
        alpha=0.14,
    )
    ax.scatter(
        [max_row["date"]],
        [max_row["total_offhire_pct"]],
        color=ACCENT_COLOR,
        s=46,
        zorder=5,
    )
    ax.annotate(
        f"Topp: {max_row['total_offhire_pct']:.1f}",
        xy=(max_row["date"], max_row["total_offhire_pct"]),
        xytext=(10, 14),
        textcoords="offset points",
        fontsize=9,
        color=TEXT_COLOR,
    )
    ax.set_title("Samlet offhire per måned")
    ax.set_xlabel("Måned")
    ax.set_ylabel("Offhire (sum av prosentpoeng)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper right")
    fig.autofmt_xdate(rotation=45)
    save_figure(fig, "samlet_offhire_per_maaned.png")


def plot_vessel_heatmap(panel_df: pd.DataFrame) -> None:
    ordered_vessels = sorted(panel_df["vessel"].unique(), key=vessel_sort_key)
    heatmap = (
        panel_df.pivot(index="vessel", columns="date", values="offhire_pct")
        .reindex(ordered_vessels)
        .sort_index(axis=1)
    )
    masked = np.ma.masked_invalid(heatmap.to_numpy())
    cmap = matplotlib.colormaps[HEATMAP_CMAP].copy()
    cmap.set_bad("#F4F7FB")

    fig, ax = plt.subplots(figsize=(14, 6.5))
    image = ax.imshow(masked, aspect="auto", cmap=cmap, vmin=0)
    ax.set_title("Heatmap for offhire per fartøy og måned")
    ax.set_ylabel("Fartøy")
    ax.set_xlabel("Måned")
    ax.set_yticks(range(len(heatmap.index)))
    ax.set_yticklabels(heatmap.index)

    month_positions = list(range(0, len(heatmap.columns), 3))
    month_labels = [heatmap.columns[index].strftime("%Y-%m") for index in month_positions]
    ax.set_xticks(month_positions)
    ax.set_xticklabels(month_labels, rotation=45, ha="right")

    colorbar = fig.colorbar(image, ax=ax, pad=0.02)
    colorbar.set_label("Offhire (%)")
    save_figure(fig, "heatmap_fartoy_maaned.png")


def plot_average_offhire_by_vessel(panel_df: pd.DataFrame) -> list[str]:
    summary = (
        panel_df.groupby("vessel")["offhire_pct"]
        .mean()
        .sort_values(ascending=True)
        .reset_index()
    )
    vessels = summary["vessel"].tolist()
    colors = [
        BAR_COLORS[index] if index < len(BAR_COLORS) else BAR_COLORS[-1]
        for index in range(len(summary))
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    ax.barh(vessels, summary["offhire_pct"], color=colors[::-1], edgecolor="white")
    ax.set_title("Gjennomsnittlig månedlig offhire per fartøy")
    ax.set_xlabel("Offhire (%)")
    ax.set_ylabel("Fartøy")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save_figure(fig, "gjennomsnitt_offhire_per_fartoy.png")

    return (
        summary.sort_values("offhire_pct", ascending=False)["vessel"].head(5).tolist()
    )


def plot_boxplot_by_vessel(panel_df: pd.DataFrame) -> None:
    ordered_vessels = (
        panel_df.groupby("vessel")["offhire_pct"]
        .mean()
        .sort_values(ascending=True)
        .index.tolist()
    )
    values = [
        panel_df.loc[panel_df["vessel"] == vessel, "offhire_pct"].to_numpy()
        for vessel in ordered_vessels
    ]

    fig, ax = plt.subplots(figsize=(10.5, 7.0))
    boxplot = ax.boxplot(
        values,
        vert=False,
        patch_artist=True,
        tick_labels=ordered_vessels,
        medianprops={"color": TEXT_COLOR, "linewidth": 1.4},
        boxprops={"linewidth": 1.0, "edgecolor": "#486581"},
        whiskerprops={"linewidth": 1.0, "color": "#486581"},
        capprops={"linewidth": 1.0, "color": "#486581"},
        flierprops={
            "marker": "o",
            "markerfacecolor": ACCENT_COLOR,
            "markeredgecolor": ACCENT_COLOR,
            "markersize": 3.5,
            "alpha": 0.65,
        },
    )
    for box in boxplot["boxes"]:
        box.set_facecolor("#A7D3D0")
        box.set_alpha(0.85)

    ax.set_title("Fordeling av offhire per fartøy")
    ax.set_xlabel("Offhire (%)")
    ax.set_ylabel("Fartøy")
    ax.grid(axis="x")
    ax.set_axisbelow(True)
    save_figure(fig, "boksplot_offhire_per_fartoy.png")


def plot_top_vessels_over_time(panel_df: pd.DataFrame, top_vessels: list[str]) -> None:
    if not top_vessels:
        return

    full_index = pd.date_range(
        start=panel_df["date"].min(),
        end=panel_df["date"].max(),
        freq="MS",
    )

    fig, ax = plt.subplots(figsize=(12, 6))
    line_colors = ["#0F4C5C", "#136F63", "#1F9D8B", "#2A6F97", "#D97706"]

    for color, vessel in zip(line_colors, top_vessels, strict=False):
        vessel_series = (
            panel_df.loc[panel_df["vessel"] == vessel, ["date", "offhire_pct"]]
            .drop_duplicates(subset=["date"])
            .set_index("date")
            .reindex(full_index)
        )
        ax.plot(
            full_index,
            vessel_series["offhire_pct"],
            label=vessel,
            linewidth=2.0,
            color=color,
        )

    ax.set_title("Tidsserier for fartøy med høyest gjennomsnittlig offhire")
    ax.set_xlabel("Måned")
    ax.set_ylabel("Offhire (%)")
    ax.grid(axis="y")
    ax.set_axisbelow(True)
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.16), ncol=3)
    fig.autofmt_xdate(rotation=45)
    save_figure(fig, "top5_fartoy_tidsserie.png")


def write_table_bundle(
    coverage_summary: pd.DataFrame,
    observations_per_year: pd.DataFrame,
    vessel_summary: pd.DataFrame,
    top_observations: pd.DataFrame,
) -> None:
    coverage_summary.to_csv(
        TABLES_DIR / "data_coverage_summary.csv",
        index=False,
        encoding="utf-8-sig",
    )
    observations_per_year.to_csv(
        TABLES_DIR / "observations_per_year.csv",
        index=False,
        encoding="utf-8-sig",
    )
    write_markdown_table(
        "data_coverage_summary.md",
        "Datadekning",
        [
            ("Oversikt", coverage_summary),
            ("Observasjoner per år", observations_per_year),
        ],
    )

    save_table(vessel_summary, "vessel_summary", "Nøkkeltall per fartøy")
    save_table(top_observations, "top10_offhire_observations", "Topp 10 offhire-observasjoner")


def write_visualization_summary(
    panel_df: pd.DataFrame,
    coverage_summary: pd.DataFrame,
    top_vessels: list[str],
) -> None:
    min_date = panel_df["date"].min()
    max_date = panel_df["date"].max()
    summary_lines = [
        "# Historiske visualiseringer",
        "",
        "Dette dokumentet oppsummerer genererte figurer og tabeller for de historiske dataene.",
        "",
        "## Datagrunnlag",
        "",
        f"- Tidsperiode: `{min_date:%Y-%m}` til `{max_date:%Y-%m}`",
        f"- Antall fartøy: `{panel_df['vessel'].nunique()}`",
        f"- Antall observasjoner: `{len(panel_df)}`",
        f"- Ufullstendige år: `{coverage_summary.loc[coverage_summary['Metrikk'] == 'Ufullstendige år', 'Verdi'].iloc[0]}`",
        "",
        "## Figurer",
        "",
        "- `figures/samlet_offhire_per_maaned.png`",
        "- `figures/heatmap_fartoy_maaned.png`",
        "- `figures/gjennomsnitt_offhire_per_fartoy.png`",
        "- `figures/boksplot_offhire_per_fartoy.png`",
        "- `figures/top5_fartoy_tidsserie.png`",
        "",
        "## Tabeller",
        "",
        "- `tables/data_coverage_summary.csv` og `tables/data_coverage_summary.md`",
        "- `tables/observations_per_year.csv`",
        "- `tables/vessel_summary.csv` og `tables/vessel_summary.md`",
        "- `tables/top10_offhire_observations.csv` og `tables/top10_offhire_observations.md`",
        "",
        "## Anbefalt bruk i rapporten",
        "",
        "- Hovedtekst: samlet offhire per måned, heatmap og datadekningstabellen.",
        "- Analyse eller vedlegg: boksplottet og tidsserien for toppfartøyene.",
        "- Nøkkeltall per fartøy kan brukes som oppslagstabell i metode- eller analysedelen.",
    ]
    if top_vessels:
        summary_lines.extend(
            [
                "",
                "## Toppfartøy i tidsseriefiguren",
                "",
                *[f"- `{vessel}`" for vessel in top_vessels],
            ]
        )

    (VISUALIZATION_DIR / "summary.md").write_text(
        "\n".join(summary_lines) + "\n",
        encoding="utf-8",
    )


def main() -> None:
    ensure_output_dirs()
    configure_matplotlib()

    panel_df, stats = load_historical_dataset()
    coverage_summary = build_data_coverage_summary(panel_df, stats)
    observations_per_year = build_observations_per_year(panel_df)
    vessel_summary = build_vessel_summary(panel_df)
    top_observations = build_top_observations(panel_df)

    plot_monthly_total_offhire(panel_df)
    plot_vessel_heatmap(panel_df)
    top_vessels = plot_average_offhire_by_vessel(panel_df)
    plot_boxplot_by_vessel(panel_df)
    plot_top_vessels_over_time(panel_df, top_vessels)

    write_table_bundle(
        coverage_summary=coverage_summary,
        observations_per_year=observations_per_year,
        vessel_summary=vessel_summary,
        top_observations=top_observations,
    )
    write_visualization_summary(
        panel_df=panel_df,
        coverage_summary=coverage_summary,
        top_vessels=top_vessels,
    )

    print("Genererte historiske visualiseringer i 004 data/visualization/.")
    print(f"- Figurer: {len(list(FIGURES_DIR.glob('*.png')))}")
    print(f"- Tabeller: {len(list(TABLES_DIR.glob('*')))}")
    print(
        f"- Datasett: {len(panel_df)} observasjoner fra "
        f"{panel_df['date'].min():%Y-%m} til {panel_df['date'].max():%Y-%m}"
    )


if __name__ == "__main__":
    main()
