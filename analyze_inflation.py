"""Clean and analyse a grocery-price CSV; generate a summary and trend chart."""
import argparse
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

REQUIRED = {"date", "country", "product", "price", "currency", "unit"}
GROUPS = ["country", "product", "currency", "unit"]

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--input", default="data/grocery_prices.csv")
    parser.add_argument("--output-dir", default="reports")
    args = parser.parse_args()

    input_path = Path(args.input)
    output_dir = Path(args.output_dir)
    if not input_path.exists():
        raise SystemExit(
            f"Dataset not found: {input_path}. See data/README.md for the required CSV format."
        )
    output_dir.mkdir(parents=True, exist_ok=True)

    df = pd.read_csv(input_path)
    missing = REQUIRED - set(df.columns)
    if missing:
        raise SystemExit(f"Missing required columns: {', '.join(sorted(missing))}")

    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df["price"] = pd.to_numeric(df["price"], errors="coerce")
    df = df.dropna(subset=["date", "country", "product", "price", "currency", "unit"])
    df = df[df["price"] > 0].copy()
    if df.empty:
        raise SystemExit("No valid rows remain after cleaning. Check the date and price columns.")

    df["month"] = df["date"].dt.to_period("M").dt.to_timestamp()
    df = df.sort_values("date")
    df.to_csv(output_dir / "cleaned_grocery_prices.csv", index=False)

    summary_rows = []
    for keys, group in df.groupby(GROUPS, dropna=False):
        group = group.sort_values("date")
        first, last = group.iloc[0], group.iloc[-1]
        change = ((last["price"] - first["price"]) / first["price"]) * 100
        summary_rows.append({
            "country": keys[0],
            "product": keys[1],
            "currency": keys[2],
            "unit": keys[3],
            "first_date": first["date"].date().isoformat(),
            "first_price": first["price"],
            "last_date": last["date"].date().isoformat(),
            "last_price": last["price"],
            "change_percent_first_to_last": round(change, 2),
            "observations": len(group),
        })
    summary = pd.DataFrame(summary_rows).sort_values(
        "change_percent_first_to_last", ascending=False
    )
    summary.to_csv(output_dir / "price_change_summary.csv", index=False)

    # Plot a manageable subset, averaging prices only within identical product/unit/currency series.
    monthly = (
        df.groupby(["month", "country", "product", "currency", "unit"], as_index=False)["price"]
        .mean()
    )
    series_keys = monthly[["country", "product", "currency", "unit"]].drop_duplicates().head(6)
    fig, ax = plt.subplots(figsize=(11, 6))
    plotted = 0
    for _, key in series_keys.iterrows():
        mask = (
            (monthly["country"] == key["country"]) &
            (monthly["product"] == key["product"]) &
            (monthly["currency"] == key["currency"]) &
            (monthly["unit"] == key["unit"])
        )
        part = monthly.loc[mask].sort_values("month")
        label = f'{key["country"]} · {key["product"]} ({key["unit"]}, {key["currency"]})'
        ax.plot(part["month"], part["price"], marker="o", linewidth=1.8, label=label)
        plotted += 1
    ax.set_title("Monthly Average Grocery Prices")
    ax.set_xlabel("Month")
    ax.set_ylabel("Price per stated unit")
    ax.grid(True, alpha=0.25)
    if plotted:
        ax.legend(fontsize=8, loc="best")
    fig.tight_layout()
    fig.savefig(output_dir / "monthly_price_trends.png", dpi=160)
    plt.close(fig)

    print(f"Valid rows analysed: {len(df)}")
    print(f"Unique product/country/unit series: {len(summary)}")
    print(f"Outputs saved to: {output_dir.resolve()}")
    print("Reminder: first-to-last price change is not an official CPI inflation rate.")

if __name__ == "__main__":
    main()
