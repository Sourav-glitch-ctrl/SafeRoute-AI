import { useState, useCallback } from "react";
import {
  MapContainer,
  TileLayer,
  Marker,
  Popup,
  useMapEvents,
  Polyline,
} from "react-leaflet";
import L from "leaflet";
import "leaflet/dist/leaflet.css";
import "./App.css";

/* ── Custom SVG pin icons ───────────────────────────────────── */

const makePin = (fill: string) =>
  L.divIcon({
    className: "",
    html: `<svg width="30" height="42" viewBox="0 0 30 42" xmlns="http://www.w3.org/2000/svg">
      <filter id="sh" x="-30%" y="-30%" width="160%" height="160%">
        <feDropShadow dx="0" dy="2" stdDeviation="2" flood-color="rgba(0,0,0,0.35)"/>
      </filter>
      <path d="M15 0C6.72 0 0 6.72 0 15c0 10.5 15 27 15 27S30 25.5 30 15C30 6.72 23.28 0 15 0z"
        fill="${fill}" filter="url(#sh)"/>
      <circle cx="15" cy="15" r="6" fill="white" opacity="0.9"/>
    </svg>`,
    iconSize: [30, 42],
    iconAnchor: [15, 42],
    popupAnchor: [0, -44],
  });

const startPin = makePin("#1a73e8");
const destPin  = makePin("#ea4335");

/* ── Types ──────────────────────────────────────────────────── */

type Location = { lat: number; lng: number };
type Step = "pick-start" | "pick-dest" | "ready" | "loading" | "result";

interface PredictionResponse {
  severity: number;
  probabilities: { [key: string]: number };
}

/* ── Severity config ─────────────────────────────────────────── */

const SEV: Record<number, { label: string; color: string; bg: string; emoji: string }> = {
  1: { label: "Low Risk",      color: "#16a34a", bg: "#dcfce7", emoji: "✅" },
  2: { label: "Moderate Risk", color: "#d97706", bg: "#fef3c7", emoji: "⚠️" },
  3: { label: "High Risk",     color: "#ea580c", bg: "#ffedd5", emoji: "🔴" },
  4: { label: "Severe Risk",   color: "#dc2626", bg: "#fee2e2", emoji: "🚨" },
};
const sevMeta = (n: number) => SEV[n] ?? { label: "Unknown", color: "#6b7280", bg: "#f3f4f6", emoji: "❓" };

/* ── Haversine ───────────────────────────────────────────────── */

function haversineMi(a: Location, b: Location) {
  const R = 3958.8;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const x = Math.sin(dLat / 2) ** 2 +
    Math.cos((a.lat * Math.PI) / 180) * Math.cos((b.lat * Math.PI) / 180) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

/* ── Map click handler ───────────────────────────────────────── */

function MapClickHandler({ step, onPick }: { step: Step; onPick: (l: Location) => void }) {
  useMapEvents({
    click(e) {
      if (step === "pick-start" || step === "pick-dest") {
        onPick({ lat: e.latlng.lat, lng: e.latlng.lng });
      }
    },
  });
  return null;
}

/* ── App ─────────────────────────────────────────────────────── */

export default function App() {
  const [start, setStart]   = useState<Location | null>(null);
  const [dest,  setDest]    = useState<Location | null>(null);
  const [step,  setStep]    = useState<Step>("pick-start");
  const [result, setResult] = useState<PredictionResponse | null>(null);
  const [error,  setError]  = useState("");
  const [weather, setWeather] = useState("Clear");
  const [hour,    setHour]    = useState(new Date().getHours());

  const handleMapPick = useCallback((loc: Location) => {
    if (step === "pick-start") { setStart(loc); setStep("pick-dest"); }
    else if (step === "pick-dest") { setDest(loc); setStep("ready"); }
  }, [step]);

  const reset = () => {
    setStart(null); setDest(null);
    setStep("pick-start"); setResult(null); setError("");
  };

  const predict = async () => {
    if (!start || !dest) return;
    setStep("loading"); setError(""); setResult(null);

    const now = new Date();
    const month = now.getMonth() + 1;
    const dow = now.getDay();
    const hasRain = weather === "Rain" || weather === "Thunderstorm";
    const isMorn = hour >= 7 && hour <= 10 ? 1 : 0;
    const isEve  = hour >= 17 && hour <= 20 ? 1 : 0;

    const body = {
      Distance_mi:          parseFloat(haversineMi(start, dest).toFixed(3)),
      Year:                 now.getFullYear(),
      Start_Lng:            start.lng,
      Start_Lat:            start.lat,
      Pressure_in:          29.92,
      Temperature_F:        75,
      Month:                month,
      Humidity_percent:     hasRain ? 85 : 55,
      Hour:                 hour,
      Wind_Speed_mph:       hasRain ? 12 : 6,
      Quarter:              Math.ceil(month / 3),
      DayOfWeek:            dow,
      Traffic_Signal:       1,
      Weather_Category:     weather,
      Visibility_mi:        weather === "Fog" ? 0.5 : weather === "Snow" ? 1.5 : 8,
      NearRoadInfrastructure: 1,
      Crossing:             0,
      Junction:             1,
      IsWeekend:            dow === 0 || dow === 6 ? 1 : 0,
      IsNight:              hour < 6 || hour >= 20 ? 1 : 0,
      IsRushHour:           isMorn || isEve,
      Precipitation_in:     hasRain ? 0.25 : 0,
      MorningRushHour:      isMorn,
      EveningRushHour:      isEve,
      Stop:                 1,
      HasPrecipitation:     hasRain ? 1 : 0,
      LowVisibility:        weather === "Fog" || weather === "Snow" ? 1 : 0,
      Railway:              0,
    };

    const ctrl = new AbortController();
    const timer = setTimeout(() => ctrl.abort(), 30000);
    try {
      const res = await fetch("/api/v1/prediction/predict", {
        method: "POST",
        headers: { "Content-Type": "application/json" },
        body: JSON.stringify(body),
        signal: ctrl.signal,
      });
      clearTimeout(timer);
      if (!res.ok) { const e = await res.json().catch(() => ({})); throw new Error(e.detail ?? `HTTP ${res.status}`); }
      setResult(await res.json());
      setStep("result");
    } catch (e: unknown) {
      clearTimeout(timer);
      setError(e instanceof DOMException && e.name === "AbortError"
        ? "Request timed out. Try again."
        : e instanceof Error ? e.message : "Could not reach backend.");
      setStep("ready");
    }
  };

  const hint =
    step === "pick-start" ? "📍 Click the map to drop your start pin" :
    step === "pick-dest"  ? "🏁 Click the map to drop your destination pin" :
    step === "ready"      ? "✨ Conditions set — ready to analyse" :
    step === "loading"    ? "🔍 Analysing route safety…" :
                            "✅ Analysis complete";

  const meta = result ? sevMeta(result.severity) : null;
  const polyColor = meta ? meta.color : "#1a73e8";
  const isPicking = step === "pick-start" || step === "pick-dest";

  return (
    <div className="app">

      {/* ── Sidebar ──────────────────────────────────── */}
      <aside className="sidebar">

        <div className="sidebar-brand">
          <div className="brand-icon">
            <svg viewBox="0 0 24 24" fill="none">
              <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="#1a73e8"/>
              <circle cx="12" cy="9" r="2.8" fill="white"/>
            </svg>
          </div>
          <div className="brand-text">
            <h1>SafeRoute AI</h1>
            <p>Road safety intelligence</p>
          </div>
        </div>

        {/* Step hint */}
        <div className={`step-pill ${step === "loading" ? "step-pill--pulse" : ""}`}>
          <span className={`step-dot step-dot--${step}`} />
          <span>{hint}</span>
        </div>

        {/* Pin rows */}
        <div className="pins-block">
          <div className={`pin-row ${start ? "pin-row--active" : ""}`}>
            <div className="pin-circle pin-circle--start">A</div>
            <div className="pin-detail">
              <span>Start point</span>
              <p>{start ? `${start.lat.toFixed(5)}, ${start.lng.toFixed(5)}` : "Click map to set"}</p>
            </div>
            {start && (
              <button className="pin-x" onClick={() => { setStart(null); setStep("pick-start"); setResult(null); }} title="Remove">✕</button>
            )}
          </div>

          <div className="pin-line" />

          <div className={`pin-row ${dest ? "pin-row--active" : ""}`}>
            <div className="pin-circle pin-circle--dest">B</div>
            <div className="pin-detail">
              <span>Destination</span>
              <p>{dest ? `${dest.lat.toFixed(5)}, ${dest.lng.toFixed(5)}` : "Click map to set"}</p>
            </div>
            {dest && (
              <button className="pin-x" onClick={() => { setDest(null); setStep(start ? "pick-dest" : "pick-start"); setResult(null); }} title="Remove">✕</button>
            )}
          </div>
        </div>

        {/* Conditions */}
        <div className="conditions">
          <p className="cond-title">Weather condition</p>
          <div className="weather-chips">
            {[
              ["Clear","☀️"], ["Cloudy","☁️"], ["Rain","🌧️"],
              ["Fog","🌫️"], ["Snow","❄️"], ["Thunderstorm","⛈️"],
            ].map(([w, icon]) => (
              <button
                key={w}
                className={`chip ${weather === w ? "chip--on" : ""}`}
                onClick={() => setWeather(w)}
              >
                {icon} {w}
              </button>
            ))}
          </div>

          <p className="cond-title" style={{ marginTop: 18 }}>
            Departure time — <strong>{String(hour).padStart(2, "0")}:00</strong>
          </p>
          <input
            type="range" min={0} max={23} value={hour}
            onChange={e => setHour(Number(e.target.value))}
            className="slider"
          />
          <div className="slider-labels">
            <span>12 AM</span><span>12 PM</span><span>11 PM</span>
          </div>
        </div>

        {/* CTA */}
        <div className="cta">
          <button
            className="btn-primary"
            disabled={step !== "ready" && step !== "result"}
            onClick={predict}
          >
            {step === "loading"
              ? <span className="spin" />
              : "Analyse Route Safety"}
          </button>
          {(start || dest) && (
            <button className="btn-ghost" onClick={reset}>Reset map</button>
          )}
        </div>

        {/* Error */}
        {error && <div className="err-box">{error}</div>}

        {/* Result */}
        {result && meta && step === "result" && (
          <div className="result-card" style={{ "--c": meta.color, "--bg": meta.bg } as React.CSSProperties}>

            <div className="result-header">
              <span className="result-emoji">{meta.emoji}</span>
              <div>
                <p>Predicted Severity</p>
                <h2 style={{ color: meta.color }}>{meta.label}</h2>
              </div>
              <div className="result-num">{result.severity}</div>
            </div>

            <div className="prob-section">
              <p className="prob-heading">Class probabilities</p>
              {Object.entries(result.probabilities)
                .sort(([a], [b]) => Number(a) - Number(b))
                .map(([sev, p]) => {
                  const m = sevMeta(Number(sev));
                  return (
                    <div key={sev} className="prob-row">
                      <span className="prob-sev" style={{ color: m.color }}>S{sev}</span>
                      <div className="prob-track">
                        <div className="prob-fill" style={{ width: `${(p * 100).toFixed(1)}%`, background: m.color }} />
                      </div>
                      <span className="prob-pct">{(p * 100).toFixed(1)}%</span>
                    </div>
                  );
                })}
            </div>

            {start && dest && (
              <div className="result-meta">
                <div className="meta-chip"><span>📏</span><strong>{haversineMi(start, dest).toFixed(2)} mi</strong></div>
                <div className="meta-chip"><span>🌤</span><strong>{weather}</strong></div>
                <div className="meta-chip"><span>🕐</span><strong>{hour}:00</strong></div>
              </div>
            )}
          </div>
        )}

      </aside>

      {/* ── Map ────────────────────────────────────────── */}
      <div className={`map-area ${isPicking ? "map-area--picking" : ""}`}>
        <MapContainer
          center={[20.59, 78.96]}
          zoom={5}
          className="lmap"
          zoomControl={false}
        >
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapClickHandler step={step} onPick={handleMapPick} />

          {start && (
            <Marker position={[start.lat, start.lng]} icon={startPin}>
              <Popup><strong>📍 Start</strong><br />{start.lat.toFixed(5)}, {start.lng.toFixed(5)}</Popup>
            </Marker>
          )}

          {dest && (
            <Marker position={[dest.lat, dest.lng]} icon={destPin}>
              <Popup><strong>🏁 Destination</strong><br />{dest.lat.toFixed(5)}, {dest.lng.toFixed(5)}</Popup>
            </Marker>
          )}

          {start && dest && (
            <Polyline
              positions={[[start.lat, start.lng], [dest.lat, dest.lng]]}
              pathOptions={{
                color: polyColor,
                weight: 5,
                opacity: 0.85,
                dashArray: step === "result" ? undefined : "12 8",
              }}
            />
          )}
        </MapContainer>

        {isPicking && (
          <div className="map-hint">
            {step === "pick-start" ? "📍 Click to set start" : "🏁 Click to set destination"}
          </div>
        )}
      </div>

    </div>
  );
}