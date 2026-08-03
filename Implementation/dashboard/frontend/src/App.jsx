import { useEffect, useState } from "react";
import HistoryPage from "./HistoryPage.jsx";
import TrendsPage from "./TrendsPage.jsx";

const fmtDate = (d) =>
  new Date(d).toLocaleDateString("en-SG", {
    day: "numeric",
    month: "short",
    year: "numeric",
  });

export default function App() {
  const [page, setPage] = useState("history"); // 'history' | 'trends'
  const [summary, setSummary] = useState(null);
  const [bounds, setBounds] = useState(null);

  useEffect(() => {
    fetch("/api/summary")
      .then((r) => r.json())
      .then(setSummary);
    fetch("/api/history")
      .then((r) => r.json())
      .then(setBounds);
  }, []);

  return (
    <div className="page">
      {/* ===== Header + navigation ===== */}
      <header className="header">
        <div>
          <h1>Singapore Electricity Demand</h1>
          <p className="subtitle">
            {page === "history"
              ? "How much electricity Singapore used each day — and how close our model came"
              : "What electricity demand is likely to look like in the days and months ahead"}
            {bounds &&
              page === "history" &&
              ` · data from ${fmtDate(bounds.min_date)} to ${fmtDate(bounds.max_date)}`}
          </p>
        </div>
        <nav className="nav-tabs">
          <button
            className={page === "history" ? "nav-tab active" : "nav-tab"}
            onClick={() => setPage("history")}
          >
            History Report
          </button>
          <button
            className={page === "trends" ? "nav-tab active" : "nav-tab"}
            onClick={() => setPage("trends")}
          >
            Demand Trends
          </button>
        </nav>
      </header>

      {page === "history" ? (
        <HistoryPage summary={summary} bounds={bounds} />
      ) : (
        <TrendsPage />
      )}

      <footer className="footer">
        SUTD Group Project · Demand forecasts by our machine-learning model
        (CatBoost-PPSO)
        {summary &&
          ` · overall accuracy ${(summary.summary.best_r2 * 100).toFixed(0)}% on unseen data`}{" "}
        · Weather data ©{" "}
        <a href="https://open-meteo.com/" target="_blank" rel="noreferrer">
          Open-Meteo.com
        </a>
      </footer>
    </div>
  );
}
