# 🚀 Quick Reference Guide

## One-Command Execution

```bash
cd /Users/tranduytai/Documents/SUTD/GroupProject/electricity-demand
python main.py
```

---

## 📁 File Locations

| File                        | Purpose       | Action                         |
| --------------------------- | ------------- | ------------------------------ |
| `data/ema_daily_demand.csv` | ⭐ Input data | Place CSV here                 |
| `configs/config.yaml`       | Settings      | Edit to customize              |
| `main.py`                   | Entry point   | `python main.py`               |
| `src/*.py`                  | Code modules  | Do NOT edit unless experienced |
| `results/`                  | Output        | Generated after running        |

---

## ⚙️ If CSV is Elsewhere

Edit `configs/config.yaml`:

```yaml
data:
  demand_csv_path: "/path/to/your/file.csv"
```

---

## 🔧 Installation (First Time Only)

```bash
# Step 1: Go to project
cd /Users/tranduytai/Documents/SUTD/GroupProject/electricity-demand

# Step 2: Create virtual environment
python3 -m venv electricity-demand

# Step 3: Activate it
source electricity-demand/bin/activate

# Step 4: Install dependencies
pip install -r requirements.txt

# Step 5: Run pipeline
python main.py
```

---

## 📊 After Running

Output saved to `results/`:

```
results/
├── model_comparison.csv      # Leaderboard
├── summary.yaml              # Summary
└── plots/                    # Visualizations
```

## 🚀 Next Run (After First Setup)

```bash
# Activate environment (if deactivated)
source venv/bin/activate

# Run pipeline
python main.py

# View results
cat results/summary.yaml
```
