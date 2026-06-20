import pandas as pd
import matplotlib.pyplot as plt

df = pd.read_csv('./results/model_comparison.csv').dropna(how='all')

# Round columns
df['MAE']  = df['MAE'].round(1)
df['RMSE'] = df['RMSE'].round(1)
df['R²']   = df['R²'].round(4)
df['MAPE'] = df['MAPE'].round(4)
df['RAE']  = df['RAE'].round(4)
df['WI']   = df['WI'].round(4)

df = df.sort_values('RMSE').reset_index(drop=True)
df.insert(0, 'Rank', range(1, len(df)+1))

name_map = {
    'catboost_ppso': 'CatBoost-PPSO',
    'linear':        'Linear Regression',
    'catboost':      'CatBoost',
    'rf':            'Random Forest',
    'xgb':           'XGBoost',
    'baseline':      'Baseline (Lag-1)',
}
df['Model'] = df['Model'].map(lambda x: name_map.get(x, x))

col_labels = ['Rank', 'Model', 'MAE', 'RMSE', 'R²', 'MAPE', 'RAE', 'WI']
cell_data  = df[col_labels].values.tolist()

fig, ax = plt.subplots(figsize=(13, 3.6))
ax.axis('off')

col_widths = [0.05, 0.22, 0.11, 0.11, 0.10, 0.10, 0.10, 0.10]

tbl = ax.table(
    cellText=cell_data,
    colLabels=col_labels,
    cellLoc='center',
    loc='center',
    colWidths=col_widths,
)
tbl.auto_set_font_size(False)
tbl.set_fontsize(10.5)
tbl.scale(1, 2.0)

HEADER_BG = '#1C2B3A'
HEADER_FG = 'white'
BEST_BG   = '#D4EFDF'
ALT_BG    = '#F4F6F9'
WHITE_BG  = 'white'
RANK1_FG  = '#1A6B3C'

n_rows = len(cell_data)

for (row, col), cell in tbl.get_celld().items():
    cell.set_edgecolor('#CBD5E0')
    cell.set_linewidth(0.6)
    if row == 0:
        cell.set_facecolor(HEADER_BG)
        cell.set_text_props(color=HEADER_FG, fontweight='bold', fontsize=11)
    elif row == 1:
        cell.set_facecolor(BEST_BG)
        cell.set_text_props(color=RANK1_FG, fontweight='bold')
    elif row % 2 == 0:
        cell.set_facecolor(ALT_BG)
    else:
        cell.set_facecolor(WHITE_BG)

for col_idx, col_name in enumerate(col_labels):
    if col_name in ('RMSE', 'R²', 'WI'):
        tbl[(1, col_idx)].set_text_props(fontweight='bold', color=RANK1_FG)

fig.suptitle(
    'Model Performance Leaderboard — Singapore Electricity Demand Forecast',
    fontsize=13, fontweight='bold', y=0.98, color='#1C2B3A'
)
fig.text(
    0.5, 0.01,
    'Sorted by RMSE (lower is better)  ·  R², WI: higher is better  ·  MAPE, RAE: lower is better',
    ha='center', fontsize=8.5, color='#718096', style='italic'
)

plt.tight_layout(rect=[0, 0.04, 1, 0.95])
out = './results/model_comparison_table.png'
plt.savefig(out, dpi=180, bbox_inches='tight', facecolor='white')
plt.close()
print(f'Saved: {out}')
