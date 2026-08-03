import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import logging
import os

logger = logging.getLogger(__name__)

# ─── Palette ────────────────────────────────────────────────────────────────
PALETTE = {
    'weekday':  '#4C9BE8',   # blue
    'weekend':  '#F28B30',   # orange
    'holiday':  '#E84C4C',   # red
    'primary':  '#4C9BE8',
    'fill':     '#A8C8F0',
    'temp':     '#F28B30',
    'humidity': '#3BAF7E',
    'solar':    '#F5C842',
}


def _day_type_colors(df: pd.DataFrame):
    """Return color array and legend patches based on is_weekend / is_holiday."""
    colors = []
    for _, row in df.iterrows():
        if row.get('is_holiday', 0) == 1:
            colors.append(PALETTE['holiday'])
        elif row.get('is_weekend', 0) == 1:
            colors.append(PALETTE['weekend'])
        else:
            colors.append(PALETTE['weekday'])

    patches = [
        mpatches.Patch(color=PALETTE['weekday'],  label='Weekday'),
        mpatches.Patch(color=PALETTE['weekend'],  label='Weekend'),
        mpatches.Patch(color=PALETTE['holiday'],  label='Public Holiday'),
    ]
    return colors, patches


# ─── 1. Time-series overview ─────────────────────────────────────────────────
def plot_time_series(df: pd.DataFrame, save_dir: str = "./results/plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    fig, axes = plt.subplots(4, 1, figsize=(15, 14), sharex=True)
    fig.suptitle('Singapore Electricity Demand & Weather – Overview',
                 fontsize=14, fontweight='bold', y=1.01)

    # ── Demand ──
    ax = axes[0]
    ax.plot(df['date'], df['demand_mwh'], linewidth=0.9, alpha=0.8,
            color=PALETTE['primary'])
    ax.fill_between(df['date'], df['demand_mwh'], alpha=0.2,
                    color=PALETTE['fill'])
    ax.set_title('Daily Electricity Demand (MWh)', fontweight='bold')
    ax.set_ylabel('MWh')
    ax.grid(True, alpha=0.3)

    # ── Temperature ──
    ax = axes[1]
    ax.plot(df['date'], df['temperature'], linewidth=0.9, alpha=0.85,
            color=PALETTE['temp'], label='Mean Temp')
    ax.set_title('Daily Mean Temperature (°C)', fontweight='bold')
    ax.set_ylabel('°C')
    ax.legend(fontsize=9)
    ax.grid(True, alpha=0.3)

    # ── Humidity ──
    ax = axes[2]
    ax.plot(df['date'], df['humidity'], linewidth=0.9, alpha=0.85,
            color=PALETTE['humidity'])
    ax.set_title('Daily Mean Relative Humidity (%)', fontweight='bold')
    ax.set_ylabel('%')
    ax.grid(True, alpha=0.3)

    # ── Solar radiation ──
    ax = axes[3]
    ax.plot(df['date'], df['solar_radiation'], linewidth=0.9, alpha=0.85,
            color=PALETTE['solar'])
    ax.fill_between(df['date'], df['solar_radiation'], alpha=0.2,
                    color=PALETTE['solar'])
    ax.set_title('Daily Shortwave Radiation Sum (MJ/m²)', fontweight='bold')
    ax.set_ylabel('MJ/m²')
    ax.set_xlabel('Date')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eda_time_series.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Time-series plot saved → {save_path}")


# ─── 2. Correlation heatmap ──────────────────────────────────────────────────
def plot_correlation_heatmap(df: pd.DataFrame, save_dir: str = "./results/plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    numeric_cols = ['temperature', 'humidity', 'wind_speed', 'solar_radiation',
                    'demand_mwh']
    # Include optional columns if present
    for col in ['is_weekend', 'is_holiday', 'day_of_week', 'month', 'cci']:
        if col in df.columns:
            numeric_cols.append(col)

    corr = df[numeric_cols].corr()

    plt.figure(figsize=(11, 9))
    mask = np.triu(np.ones_like(corr, dtype=bool), k=1)  # show full matrix
    sns.heatmap(corr, annot=True, fmt='.2f', cmap='coolwarm', center=0,
                vmin=-1, vmax=1, linewidths=0.5,
                cbar_kws={'label': 'Pearson r', 'shrink': 0.8})
    plt.title('Correlation Matrix – Weather Variables vs Electricity Demand',
              fontsize=13, fontweight='bold', pad=12)
    plt.xticks(rotation=30, ha='right', fontsize=9)
    plt.yticks(fontsize=9)
    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eda_correlation_heatmap.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Correlation heatmap saved → {save_path}")


# ─── 3. Scatter plots (coloured by day type) ────────────────────────────────
def plot_scatter_demand(df: pd.DataFrame, save_dir: str = "./results/plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Ensure required columns exist
    for col in ['is_weekend', 'is_holiday']:
        if col not in df.columns:
            df[col] = 0

    colors, patches = _day_type_colors(df)

    fig, axes = plt.subplots(1, 2, figsize=(15, 6))
    fig.suptitle('Weather vs Electricity Demand  (colour = day type)',
                 fontsize=13, fontweight='bold')

    # ── Temperature vs Demand ──
    ax = axes[0]
    ax.scatter(df['temperature'], df['demand_mwh'],
               c=colors, alpha=0.6, s=28, edgecolors='none')
    ax.set_xlabel('Mean Temperature (°C)', fontsize=11)
    ax.set_ylabel('Electricity Demand (MWh)', fontsize=11)
    ax.set_title('Temperature vs Demand', fontweight='bold')
    ax.legend(handles=patches, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    # ── Humidity vs Demand ──
    ax = axes[1]
    ax.scatter(df['humidity'], df['demand_mwh'],
               c=colors, alpha=0.6, s=28, edgecolors='none')
    ax.set_xlabel('Mean Relative Humidity (%)', fontsize=11)
    ax.set_ylabel('Electricity Demand (MWh)', fontsize=11)
    ax.set_title('Humidity vs Demand', fontweight='bold')
    ax.legend(handles=patches, fontsize=9, loc='upper left')
    ax.grid(True, alpha=0.25)

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eda_scatter_demand.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Scatter plot saved → {save_path}")


# ─── 4. Demand by day-of-week & holiday status (boxplots) ───────────────────
def plot_demand_by_daytype(df: pd.DataFrame, save_dir: str = "./results/plots") -> None:
    os.makedirs(save_dir, exist_ok=True)

    # Ensure required columns exist
    for col in ['day_of_week', 'is_holiday']:
        if col not in df.columns:
            if col == 'day_of_week':
                df[col] = df['date'].dt.dayofweek
            else:
                df[col] = 0

    day_labels = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat', 'Sun']

    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    fig.suptitle('Demand Distribution by Day Type', fontsize=13, fontweight='bold')

    # ── Boxplot by day of week (seaborn for clean palette) ──
    ax = axes[0]
    day_colors = [PALETTE['weekday']] * 5 + [PALETTE['weekend']] * 2
    df_plot = df.copy()
    df_plot['day_name'] = df_plot['day_of_week'].map(
        dict(enumerate(day_labels)))
    df_plot['day_name'] = pd.Categorical(df_plot['day_name'],
                                          categories=day_labels, ordered=True)
    sns.boxplot(data=df_plot, x='day_name', y='demand_mwh',
                palette=day_colors, ax=ax, width=0.55, fliersize=3,
                order=day_labels)
    ax.set_xlabel('Day of Week', fontsize=11)
    ax.set_ylabel('Demand (MWh)', fontsize=11)
    ax.set_title('Demand by Day of Week', fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')
    # legend
    ax.legend(handles=[
        mpatches.Patch(color=PALETTE['weekday'], label='Weekday'),
        mpatches.Patch(color=PALETTE['weekend'], label='Weekend'),
    ], fontsize=9)

    # ── Boxplot: Holiday vs Regular ──
    ax = axes[1]
    df_plot['Day Status'] = df_plot['is_holiday'].map(
        {0: 'Regular Day', 1: 'Public Holiday'})
    sns.boxplot(data=df_plot, x='Day Status', y='demand_mwh',
                palette={'Regular Day': PALETTE['weekday'],
                         'Public Holiday': PALETTE['holiday']},
                ax=ax, width=0.45, fliersize=3)
    ax.set_xlabel('')
    ax.set_ylabel('Demand (MWh)', fontsize=11)
    ax.set_title('Holiday vs Regular Days', fontweight='bold')
    ax.grid(True, alpha=0.25, axis='y')

    plt.tight_layout()
    save_path = os.path.join(save_dir, 'eda_demand_by_daytype.png')
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    plt.close()
    logger.info(f"✓ Day-type boxplot saved → {save_path}")


# ─── 5. Summary statistics (console) ────────────────────────────────────────
def print_eda_summary(df: pd.DataFrame) -> None:
    print(f"\n{'='*55}")
    print("EDA SUMMARY STATISTICS")
    print(f"{'='*55}")
    print(f"  Date range  : {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"  Total days  : {len(df)}")
    print(f"\n  Demand (MWh):")
    d = df['demand_mwh']
    print(f"    Mean={d.mean():.0f}  Std={d.std():.0f}  "
          f"Min={d.min():.0f}  Max={d.max():.0f}")

    if 'is_weekend' in df.columns:
        print(f"\n  By Day Status:")
        grp = df.groupby(df['date'].dt.dayofweek < 5)['demand_mwh'].mean()
        print(f"    Weekday avg : {grp.get(True, float('nan')):.0f} MWh")
        print(f"    Weekend avg : {grp.get(False, float('nan')):.0f} MWh")

    if 'is_holiday' in df.columns:
        grp2 = df.groupby('is_holiday')['demand_mwh'].mean()
        print(f"    Holiday avg : {grp2.get(1, float('nan')):.0f} MWh")
    print(f"{'='*55}\n")


# ─── Master EDA runner ───────────────────────────────────────────────────────
def run_eda(df: pd.DataFrame, save_dir: str = "./results/plots") -> None:
    """Run all EDA plots on the raw merged dataframe (after load_all_data)."""
    logger.info("Running EDA visualizations...")
    plot_time_series(df, save_dir)
    plot_correlation_heatmap(df, save_dir)
    plot_scatter_demand(df, save_dir)
    plot_demand_by_daytype(df, save_dir)
    print_eda_summary(df)
    logger.info("✓ All EDA plots saved to: %s", save_dir)
