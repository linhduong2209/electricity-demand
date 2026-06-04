import sys, pathlib, subprocess
from io import BytesIO

# Auto-install dependencies
for _imp, _pkg in [('pandas', 'pandas'), ('xlrd', 'xlrd')]:
    try:
        __import__(_imp)
    except ImportError:
        print(f"Installing {_pkg}...")
        subprocess.check_call([sys.executable, '-m', 'pip', 'install',
                               _pkg, '-q', '--break-system-packages'])

import pandas as pd


def parse_weekly_xls(path: pathlib.Path) -> pd.DataFrame:
    """
    Parse one EMA weekly XLS → DataFrame[date, demand_mwh].

    Layout:
      Row 1 : dates at cols 1, 4, 7, 10, 13, 16, 19
      Row 5+: 48 half-hourly MW readings per day
    Daily MWh = sum(MW readings) × 0.5
    """
    df = pd.read_excel(BytesIO(path.read_bytes()), engine="xlrd", header=None)

    date_cols = list(range(1, df.shape[1], 3))
    dates = []
    for c in date_cols:
        val = df.iloc[1, c]
        if pd.notna(val):
            try:
                dates.append(pd.to_datetime(val).date())
            except Exception:
                pass

    data_rows = df.iloc[5:5 + 48]
    records = []
    for col_idx, date in zip(date_cols, dates):
        system_mw = pd.to_numeric(data_rows.iloc[:, col_idx], errors="coerce")
        if system_mw.isna().all():
            continue
        records.append({
            "date":       pd.to_datetime(date),
            "demand_mwh": round(system_mw.sum() * 0.5)
        })
    return pd.DataFrame(records)


def build_csv(root: str, output_csv: str = "ema_daily_demand.csv") -> pd.DataFrame:
    root_path = pathlib.Path(root)
    if not root_path.exists():
        sys.exit(f"Folder not found: '{root}'")

    # rglob finds all .xls files at any nesting depth: root/year/month/file.xls
    xls_files = sorted(root_path.rglob("*.xls*"))
    if not xls_files:
        sys.exit(f"No .xls files found under '{root}'")

    print(f"Found {len(xls_files)} XLS files under '{root}'\n")

    frames = []
    errors = []

    for f in xls_files:
        rel = f.relative_to(root_path)
        try:
            week_df = parse_weekly_xls(f)
            frames.append(week_df)
            date_range = (f"{week_df['date'].min().date()} → "
                          f"{week_df['date'].max().date()}")
            print(f"  ✓  {str(rel):<40}  {len(week_df)} days  [{date_range}]")
        except Exception as e:
            errors.append((str(rel), str(e)))
            print(f"  ✗  {str(rel):<40}  ERROR: {e}")

    if not frames:
        sys.exit("No files parsed successfully.")

    demand_df = (
        pd.concat(frames, ignore_index=True)
        .drop_duplicates("date")
        .sort_values("date")
        .reset_index(drop=True)
    )

    demand_df.to_csv(output_csv, index=True)

    print(f"\n{'='*60}")
    print(f"  Total daily rows : {len(demand_df)}")
    print(f"  Date range       : {demand_df['date'].min().date()} → {demand_df['date'].max().date()}")
    print(f"  Demand (MWh)     : min={demand_df['demand_mwh'].min():,}  "
          f"max={demand_df['demand_mwh'].max():,}  "
          f"mean={demand_df['demand_mwh'].mean():,.0f}")
    if errors:
        print(f"  Skipped files    : {len(errors)}")
    print(f"  Saved to         : {output_csv}")
    print(f"{'='*60}")
    print(f"\nSample output:")
    print(demand_df.head(10).to_string())
    return demand_df


if __name__ == "__main__":
    if len(sys.argv) < 2:
        print(__doc__)
        sys.exit(0)

    src = sys.argv[1]
    out = sys.argv[2] if len(sys.argv) > 2 else "ema_daily_demand.csv"
    build_csv(src, out)