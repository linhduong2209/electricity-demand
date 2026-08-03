import { useEffect, useMemo, useState } from "react";
import {
  ResponsiveContainer,
  ComposedChart,
  Line,
  Area,
  Bar,
  XAxis,
  YAxis,
  Tooltip,
  CartesianGrid,
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

const COLORS = {
  primary: "#2563eb",
  accent: "#f59e0b",
  temp: "#dc2626",
  rain: "#0284c7",
  humidity: "#7c3aed",
  grid: "#e5e7eb",
};

const SEASON_NAMES = {
  1: "Jan – Mar",
  2: "Apr – Jun",
  3: "Jul – Sep",
  4: "Oct – Dec",
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

function accuracyBadge(daysAhead) {
  if (daysAhead <= 3) return { text: "High confidence (85–95%)", cls: "good" };
  if (daysAhead <= 7) return { text: "Good confidence (70–80%)", cls: "ok" };
  if (daysAhead <= 16) return { text: "Indicative only (50–60%)", cls: "low" };
  return { text: "Long-range trend — indicative only", cls: "low" };
}

export default function TrendsPage() {
  const now = new Date();
  const curYM = `${now.getFullYear()}-${String(now.getMonth() + 1).padStart(2, "0")}`;

  const [mode, setMode] = useState("days");
  const [days, setDays] = useState(7);
  const [month, setMonth] = useState(curYM);
  const [season, setSeason] = useState(
    String(Math.floor(now.getMonth() / 3) + 1),
  );
  const [year, setYear] = useState(String(now.getFullYear()));
  const [weatherModel, setWeatherModel] = useState("best_match");

  const [data, setData] = useState(null);
  const [err, setErr] = useState(null);
  const [loading, setLoading] = useState(false);

  const query = useMemo(() => {
    const p = new URLSearchParams({ mode, weather_model: weatherModel });
    if (mode === "days") p.set("days", days);
    if (mode === "month") p.set("month", month);
    if (mode === "season") {
      p.set("season", season);
      p.set("year", year);
    }
    return p.toString();
  }, [mode, days, month, season, year, weatherModel]);

  useEffect(() => {
    setLoading(true);
    setErr(null);
    fetch(`/api/forecast?${query}`)
      .then(async (r) => {
        const j = await r.json();
        if (!r.ok) throw new Error(j.detail || "Forecast failed");
        setData(j);
      })
      .catch((e) => {
        setErr(e.message);
        setData(null);
      })
      .finally(() => setLoading(false));
  }, [query]);

  const st = data?.stats;
  const longRange = mode !== "days";
  const maxAhead = data ? Math.max(...data.series.map((s) => s.days_ahead)) : 0;
  const badge = accuracyBadge(maxAhead);

  return (
    <>
      {/* ===== Period selector ===== */}
      <section className="card selector-card">
        <div className="selector-tabs">
          {[
            ["days", "Next days"],
            ["month", "Month"],
            ["season", "Season"],
          ].map(([m, lbl]) => (
            <button
              key={m}
              className={mode === m ? "tab active" : "tab"}
              onClick={() => setMode(m)}
            >
              {lbl}
            </button>
          ))}
        </div>
        <div className="selector-inputs">
          {mode === "days" && (
            <label>
              Horizon
              <select
                value={days}
                onChange={(e) => setDays(Number(e.target.value))}
              >
                <option value={7}>Next 7 days</option>
                <option value={16}>Next 16 days</option>
              </select>
            </label>
          )}
          {mode === "month" && (
            <label>
              Month
              <input
                type="month"
                value={month}
                min={curYM}
                onChange={(e) => setMonth(e.target.value)}
              />
            </label>
          )}
          {mode === "season" && (
            <>
              <label>
                Season
                <select
                  value={season}
                  onChange={(e) => setSeason(e.target.value)}
                >
                  {[1, 2, 3, 4].map((s) => (
                    <option key={s} value={s}>
                      {SEASON_NAMES[s]}
                    </option>
                  ))}
                </select>
              </label>
              <label>
                Year
                <select value={year} onChange={(e) => setYear(e.target.value)}>
                  {[now.getFullYear(), now.getFullYear() + 1].map((y) => (
                    <option key={y} value={y}>
                      {y}
                    </option>
                  ))}
                </select>
              </label>
            </>
          )}
          {mode === "days" && (
            <label>
              Weather model
              <select
                value={weatherModel}
                onChange={(e) => setWeatherModel(e.target.value)}
              >
                <option value="best_match">Auto (best match)</option>
                <option value="ecmwf_ifs025">
                  ECMWF (Europe, highest accuracy)
                </option>
              </select>
            </label>
          )}
          {data && (
            <span className="period-label">
              Showing:{" "}
              <strong>
                {fmtDate(data.start)} – {fmtDate(data.end)}
              </strong>
            </span>
          )}
        </div>
        {err && <p className="error">{err}</p>}
      </section>

      {/* ===== Confidence banner ===== */}
      <section className={`banner ${badge.cls}`}>
        <strong>{badge.text}.</strong> Weather forecasts are provided by{" "}
        <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">
          Open-Meteo
        </a>
        . Accuracy is very high for the first 1–3 days (85–95%), good at 4–7
        days (70–80%), and indicative beyond 8 days (50–60%).
        {longRange &&
          " Long-range monthly/seasonal outlooks use a seasonal climate model and show the general trend only."}{" "}
        Electricity demand estimates also become less precise the further ahead
        we look, because each day's forecast builds on the previous one.
      </section>

      {/* ===== Metric cards ===== */}
      <section className="metrics-row">
        <MetricCard
          label="Expected daily use"
          value={st ? `${fmt(st.avg_demand)} MWh` : "—"}
          sub={st ? `average across ${st.days} days` : ""}
        />
        <MetricCard
          label="Expected busiest day"
          value={st ? `${fmt(st.peak_demand)} MWh` : "—"}
          sub={st ? fmtDate(st.peak_date) : ""}
          tone="warm"
        />
        <MetricCard
          label="Expected quietest day"
          value={st ? `${fmt(st.low_demand)} MWh` : "—"}
          sub={st ? fmtDate(st.low_date) : ""}
          tone="cool"
        />
        <MetricCard
          label="Expected temperature"
          value={st ? `${st.avg_temperature} °C` : "—"}
          sub="period average"
          tone="good"
        />
      </section>

      {/* ===== Main chart ===== */}
      <section className="card chart-card">
        <h2>
          {longRange
            ? "Expected electricity demand trend"
            : "Expected daily electricity demand"}
        </h2>
        <p className="chart-desc">
          {longRange
            ? "The blue band shows the day-by-day estimate; the solid line is the smoothed 7-day trend, which is the most reliable signal at this range."
            : "Estimated electricity use for each day, based on the Open-Meteo weather forecast and Singapore\u2019s recent demand pattern."}
        </p>
        {loading ? (
          <div className="loading">Building forecast…</div>
        ) : (
          <ResponsiveContainer width="100%" height={320}>
            <ComposedChart
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
                formatter={(v, name) => [`${fmt(v)} MWh`, name]}
                labelFormatter={fmtDate}
              />
              <Legend />
              {longRange ? (
                <>
                  <Area
                    type="monotone"
                    dataKey="predicted_mwh"
                    name="Daily estimate"
                    stroke="none"
                    fill={COLORS.primary}
                    fillOpacity={0.15}
                  />
                  <Line
                    type="monotone"
                    dataKey="trend_mwh"
                    name="7-day trend"
                    stroke={COLORS.primary}
                    dot={false}
                    strokeWidth={2.2}
                  />
                </>
              ) : (
                <Line
                  type="monotone"
                  dataKey="predicted_mwh"
                  name="Expected demand"
                  stroke={COLORS.primary}
                  strokeWidth={2}
                  dot={{ r: 3 }}
                />
              )}
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </section>

      {/* ===== Weather chart ===== */}
      <section className="card chart-card" style={{ marginTop: 14 }}>
        <h2>Expected weather (drives the demand forecast)</h2>
        <p className="chart-desc">
          Warmer days generally mean more air-conditioning and higher
          electricity use. Weather data:{" "}
          <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">
            Open-Meteo
          </a>
          {mode === "days"
            ? ` (${weatherModel === "ecmwf_ifs025" ? "ECMWF model" : "best-match model"}).`
            : " seasonal climate model."}
        </p>
        {loading ? (
          <div className="loading">Loading…</div>
        ) : (
          <ResponsiveContainer width="100%" height={220}>
            <ComposedChart
              data={data?.series || []}
              margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
            >
              <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
              <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={60} />
              <YAxis
                tick={{ fontSize: 11 }}
                domain={["auto", "auto"]}
                unit="°C"
              />
              <Tooltip labelFormatter={fmtDate} />
              <Legend />
              {mode === "days" && (
                <Area
                  type="monotone"
                  dataKey="temperature_max"
                  name="Max temp (°C)"
                  stroke="none"
                  fill={COLORS.temp}
                  fillOpacity={0.12}
                />
              )}
              <Line
                type="monotone"
                dataKey="temperature"
                name="Avg temp (°C)"
                stroke={COLORS.temp}
                dot={false}
                strokeWidth={2}
              />
            </ComposedChart>
          </ResponsiveContainer>
        )}
      </section>

      {/* ===== Rainfall + humidity charts ===== */}
      <section className="grid-bottom two-col" style={{ marginTop: 14 }}>
        <div className="card chart-card">
          <h2>Expected rainfall</h2>
          <p className="chart-desc">
            Daily precipitation forecast (mm). Higher rainfall can reduce
            cooling demand on some days.
          </p>
          {loading ? (
            <div className="loading" style={{ height: 220 }}>
              Loading…
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart
                data={data?.series || []}
                margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
              >
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={60} />
                <YAxis
                  tick={{ fontSize: 11 }}
                  domain={["auto", "auto"]}
                  label={{
                    value: "mm",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  formatter={(v) => [
                    v == null ? "—" : `${Number(v).toFixed(1)} mm`,
                    "Rainfall",
                  ]}
                  labelFormatter={fmtDate}
                />
                <Legend />
                <Bar
                  dataKey="precipitation"
                  name="Rainfall (mm)"
                  fill={COLORS.rain}
                  radius={[5, 5, 0, 0]}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>

        <div className="card chart-card">
          <h2>Expected humidity</h2>
          <p className="chart-desc">
            Relative humidity forecast (%). Combined with temperature, this
            affects thermal comfort and cooling needs.
          </p>
          {loading ? (
            <div className="loading" style={{ height: 220 }}>
              Loading…
            </div>
          ) : (
            <ResponsiveContainer width="100%" height={220}>
              <ComposedChart
                data={data?.series || []}
                margin={{ top: 8, right: 16, left: 8, bottom: 0 }}
              >
                <CartesianGrid stroke={COLORS.grid} strokeDasharray="3 3" />
                <XAxis dataKey="date" tick={{ fontSize: 11 }} minTickGap={60} />
                <YAxis
                  tick={{ fontSize: 11 }}
                  domain={["auto", "auto"]}
                  label={{
                    value: "%",
                    angle: -90,
                    position: "insideLeft",
                    fontSize: 11,
                  }}
                />
                <Tooltip
                  formatter={(v) => [
                    v == null ? "—" : `${Number(v).toFixed(0)}%`,
                    "Humidity",
                  ]}
                  labelFormatter={fmtDate}
                />
                <Legend />
                <Line
                  type="monotone"
                  dataKey="humidity"
                  name="Humidity (%)"
                  stroke={COLORS.humidity}
                  dot={false}
                  strokeWidth={2}
                />
              </ComposedChart>
            </ResponsiveContainer>
          )}
        </div>
      </section>
    </>
  );
}
