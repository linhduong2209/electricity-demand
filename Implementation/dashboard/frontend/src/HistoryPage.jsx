import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  LineChart,
  Line,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
  BarChart,
  Bar,
  Cell,
  Legend,
} from "recharts";

const fmt = (n) =>
  n == null
    ? "—"
    : Number(n).toLocaleString("en-SG", { maximumFractionDigits: 0 });

const fmtDate = (d) =>
  new Date(d).toLocaleDateString("en-SG", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

const DOW = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"];

const COLORS = {
  primary: "#2563eb",
  accent: "#f59e0b",
  actual: "#0f766e",
  grid: "#e5e7eb",
};

const FRIENDLY_FEATURES = {
  demand_lag_1: "Yesterday's demand",
  demand_lag_7: "Demand 1 week ago",
  demand_rolling_mean_3: "3-day average demand",
  demand_rolling_mean_7: "7-day average demand",
  demand_rolling_std_3: "3-day demand variability",
  demand_rolling_std_7: "7-day demand variability",
  temperature: "Temperature",
  humidity: "Humidity",
  wind_speed: "Wind speed",
  solar_radiation: "Sunshine",
  cci: "Comfort index (heat + humidity)",
  day_of_week: "Day of week",
  month: "Month of year",
  season: "Season",
  is_weekend: "Weekend",
  is_holiday: "Public holiday",
};

function MetricCard({ label, value, sub, tone }) {
  return (
    <div className={`card metric ${tone || ""}`}>
      <span className="metric-label">{label}</span>
      <span className="metric-value">{value}</span>
      {sub && <span className="metric-sub">{sub}</span>}
    </div>
  );
}

function monthRange(ym) {
  const [y, m] = ym.split("-").map(Number);
  const last = new Date(y, m, 0).getDate();
  return [`${ym}-01`, `${ym}-${String(last).padStart(2, "0")}`];
}

export default function HistoryPage({ summary, bounds }) {
  const [mode, setMode] = useState("range");
  const [rangeStart, setRangeStart] = useState("2025-01-01");
  const [rangeEnd, setRangeEnd] = useState("2025-03-31");
  const [month, setMonth] = useState("2025-10");
  const [year, setYear] = useState("2025");

  const [data, setData] = useState(null);
  const [rangeErr, setRangeErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const [start, end] = useMemo(() => {
    if (mode === "range") return [rangeStart, rangeEnd];
    if (mode === "month") return monthRange(month);
    return [`${year}-01-01`, `${year}-12-31`];
  }, [mode, rangeStart, rangeEnd, month, year]);

  const validation = useMemo(() => {
    if (!start || !end) return "Please select a period.";
    const d0 = new Date(start),
      d1 = new Date(end);
    if (d1 < d0) return "End date must be after start date.";
    const days = (d1 - d0) / 86400000 + 1;
    if (days < 30)
      return `The period must cover at least 30 days (currently ${Math.round(days)}).`;
    return null;
  }, [start, end]);

  useEffect(() => {
    if (validation) return;
    setLoading(true);
    setRangeErr(null);
    fetch(`/api/range?start=${start}&end=${end}`)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || "Failed to load period data");
        setData(j);
      })
      .catch((e) => {
        setRangeErr(e.message);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [start, end, validation]);

  const dowData = useMemo(() => {
    if (!data) return [];
    return data.avg_by_dow.map((v, i) => ({ day: DOW[i], demand: v }));
  }, [data]);

  const fiData = useMemo(() => {
    if (!summary) return [];
    return summary.feature_importance.slice(0, 8).map((f) => ({
      ...f,
      feature: FRIENDLY_FEATURES[f.feature] || f.feature,
    }));
  }, [summary]);

  const st = data?.stats;

  return (
    <>
      {/* ===== Period selector ===== */}
      <section className="card selector-card">
        <div className="selector-tabs">
          {["range", "month", "year"].map((m) => (
            <button
              key={m}
              className={mode === m ? "tab active" : "tab"}
              onClick={() => setMode(m)}
            >
              {m === "range" ? "Date range" : m === "month" ? "Month" : "Year"}
            </button>
          ))}
        </div>
        <div className="selector-inputs">
          {mode === "range" && (
            <>
              <label>
                From
                <input
                  type="date"
                  value={rangeStart}
                  min={bounds?.min_date}
                  max={bounds?.max_date}
                  onChange={(e) => setRangeStart(e.target.value)}
                />
              </label>
              <label>
                To
                <input
                  type="date"
                  value={rangeEnd}
                  min={bounds?.min_date}
                  max={bounds?.max_date}
                  onChange={(e) => setRangeEnd(e.target.value)}
                />
              </label>
            </>
          )}
          {mode === "month" && (
            <label>
              Month
              <input
                type="month"
                value={month}
                min={bounds?.min_date?.slice(0, 7)}
                max={bounds?.max_date?.slice(0, 7)}
                onChange={(e) => setMonth(e.target.value)}
              />
            </label>
          )}
          {mode === "year" && (
            <label>
              Year
              <select value={year} onChange={(e) => setYear(e.target.value)}>
                {["2023", "2024", "2025"].map((y) => (
                  <option key={y}>{y}</option>
                ))}
              </select>
            </label>
          )}
          <span className="period-label">
            Showing:{" "}
            <strong>
              {fmtDate(start)} – {fmtDate(end)}
            </strong>
          </span>
        </div>
        {(validation || rangeErr) && (
          <p className="error">{validation || rangeErr}</p>
        )}
      </section>

      {/* ===== Metric cards ===== */}
      <section className="metrics-row">
        <MetricCard
          label="Average daily use"
          value={st ? `${fmt(st.avg_demand)} MWh` : "—"}
          sub={st ? `across ${st.days} days` : ""}
        />
        <MetricCard
          label="Busiest day"
          value={st ? `${fmt(st.peak_demand)} MWh` : "—"}
          sub={st ? fmtDate(st.peak_date) : ""}
          tone="warm"
        />
        <MetricCard
          label="Quietest day"
          value={st ? `${fmt(st.low_demand)} MWh` : "—"}
          sub={st ? fmtDate(st.low_date) : ""}
          tone="cool"
        />
        <MetricCard
          label="Forecast accuracy"
          value={st ? `${st.accuracy_pct}%` : "—"}
          sub="how close our forecasts were in this period"
          tone="good"
        />
      </section>

      {/* ===== Main chart ===== */}
      <section className="card chart-card">
        <h2>Daily electricity use — actual vs forecast</h2>
        <p className="chart-desc">
          The green line shows the real electricity used each day; the orange
          line is what our model forecast for the same day.
        </p>
        {loading ? (
          <div className="loading">Loading…</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <LineChart
              data={data?.series || []}
              margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
            >
              <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={60} />
              <YAxis
                tick={{ fontSize: 11 }}
                domain={["auto", "auto"]}
                tickFormatter={(v) => `${Math.round(v / 1000)}k`}
                label={{
                  value: "MWh",
                  angle: -90,
                  position: "insideLeft",
                  fontSize: 11,
                }}
              />
              <Tooltip
                formatter={(v) => `${fmt(v)} MWh`}
                labelFormatter={fmtDate}
              />
              <Legend />
              <Line
                type="monotone"
                dataKey="demand_mwh"
                name="Actual"
                stroke={COLORS.actual}
                dot={false}
                strokeWidth={1.8}
              />
              <Line
                type="monotone"
                dataKey="predicted_mwh"
                name="Forecast"
                stroke={COLORS.accent}
                dot={false}
                strokeWidth={1.6}
              />
            </LineChart>
          </ResponsiveContainer>
        )}
      </section>

      {/* ===== Bottom row ===== */}
      <section className="grid-bottom two-col">
        <div className="card chart-card">
          <h2>Typical use by day of the week</h2>
          <p className="chart-desc">
            {st && st.weekend_avg < st.weekday_avg
              ? `Weekends are quieter — about ${fmt(st.weekday_avg - st.weekend_avg)} MWh less than weekdays on average.`
              : "Average electricity use for each day of the week in this period."}
          </p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              data={dowData}
              margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
            >
              <CartesianGrid
                stroke={COLORS.grid}
                strokeDasharray="3 3"
                vertical={false}
              />
              <XAxis dataKey="day" tick={{ fontSize: 12 }} />
              <YAxis
                tick={{ fontSize: 11 }}
                domain={["auto", "auto"]}
                tickFormatter={(v) => `${Math.round(v / 1000)}k`}
              />
              <Tooltip formatter={(v) => `${fmt(v)} MWh`} />
              <Bar dataKey="demand" radius={[6, 6, 0, 0]}>
                {dowData.map((d, i) => (
                  <Cell
                    key={i}
                    fill={i >= 5 ? COLORS.accent : COLORS.primary}
                  />
                ))}
              </Bar>
            </BarChart>
          </ResponsiveContainer>
        </div>

        <div className="card chart-card">
          <h2>What drives electricity use?</h2>
          <p className="chart-desc">
            The factors our model relies on most. Recent demand matters most,
            followed by the calendar and the weather.
          </p>
          <ResponsiveContainer width="100%" height={230}>
            <BarChart
              data={fiData}
              layout="vertical"
              margin={{ top: 8, right: 24, left: 10, bottom: 0 }}
            >
              <CartesianGrid
                stroke={COLORS.grid}
                strokeDasharray="3 3"
                horizontal={false}
              />
              <XAxis type="number" tick={{ fontSize: 11 }} hide />
              <YAxis
                type="category"
                dataKey="feature"
                width={175}
                tick={{ fontSize: 11 }}
              />
              <Tooltip formatter={(v) => [`${v}`, "Influence"]} />
              <Bar
                dataKey="importance"
                fill={COLORS.primary}
                radius={[0, 6, 6, 0]}
              />
            </BarChart>
          </ResponsiveContainer>
        </div>
      </section>
    </>
  );
}
