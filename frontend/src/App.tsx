import { useState, useCallback, useRef, useEffect } from "react";
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

/* -- Custom SVG pin icons ------------------------------------- */

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

/* -- Types ---------------------------------------------------- */

type Location   = { lat: number; lng: number; label?: string };
type PickTarget = "start" | "dest" | null;

interface NominatimResult {
  place_id: number;
  display_name: string;
  lat: string;
  lon: string;
}

interface PredictionResponse {
  severity: number;
  probabilities: { [key: string]: number };
}

interface RouteResponse {
  coordinates: [number, number][];
  distance: number;
  duration: number;
}

/* -- Quick city presets --------------------------------------- */

const QUICK_CITIES: Location[] = [
  { lat: 28.6139, lng: 77.2090, label: "New Delhi"  },
  { lat: 19.0760, lng: 72.8777, label: "Mumbai"     },
  { lat: 12.9716, lng: 77.5946, label: "Bangalore"  },
  { lat: 13.0827, lng: 80.2707, label: "Chennai"    },
  { lat: 22.5726, lng: 88.3639, label: "Kolkata"    },
  { lat: 17.3850, lng: 78.4867, label: "Hyderabad"  },
  { lat: 23.0225, lng: 72.5714, label: "Ahmedabad"  },
  { lat: 18.5204, lng: 73.8567, label: "Pune"       },
  { lat: 26.9124, lng: 75.7873, label: "Jaipur"     },
  { lat: 30.7333, lng: 76.7794, label: "Chandigarh" },
];

/* -- Location search input ------------------------------------- */

function LocationSearch({
  placeholder,
  value,
  onSelect,
  isDropping,
  onDropper,
}: {
  placeholder: string;
  value: Location | null;
  onSelect: (loc: Location) => void;
  isDropping: boolean;
  onDropper: () => void;
}) {
  const [query, setQuery]     = useState(value?.label ?? "");
  const [results, setResults] = useState<NominatimResult[]>([]);
  const [open, setOpen]       = useState(false);
  const [busy, setBusy]       = useState(false);
  const timerRef              = useRef<ReturnType<typeof setTimeout> | null>(null);
  const wrapRef               = useRef<HTMLDivElement>(null);

  /* Sync when external value changes (map click / city preset) */
  useEffect(() => {
    setQuery(value?.label ?? (value ? `${value.lat.toFixed(5)}, ${value.lng.toFixed(5)}` : ""));
    setResults([]);
    setOpen(false);
  }, [value]);

  /* Close on outside click */
  useEffect(() => {
    const h = (e: MouseEvent) => {
      if (wrapRef.current && !wrapRef.current.contains(e.target as Node)) setOpen(false);
    };
    document.addEventListener("mousedown", h);
    return () => document.removeEventListener("mousedown", h);
  }, []);

  const search = (q: string) => {
    setQuery(q);
    if (timerRef.current) clearTimeout(timerRef.current);
    if (!q.trim()) { setResults([]); setOpen(false); return; }
    timerRef.current = setTimeout(async () => {
      setBusy(true);
      try {
        const res  = await fetch(
          `https://nominatim.openstreetmap.org/search?format=json&limit=5&q=${encodeURIComponent(q)}`,
          { headers: { "Accept-Language": "en" } }
        );
        const data: NominatimResult[] = await res.json();
        setResults(data);
        setOpen(data.length > 0);
      } catch { /* ignore */ } finally { setBusy(false); }
    }, 350);
  };

  const pick = (r: NominatimResult) => {
    const label = r.display_name.split(",").slice(0, 3).join(",").trim();
    setQuery(label);
    setResults([]);
    setOpen(false);
    onSelect({ lat: parseFloat(r.lat), lng: parseFloat(r.lon), label });
  };

  return (
    <div className="loc-search" ref={wrapRef}>
      <div className="loc-input-wrap">
        <input
          className="loc-input"
          type="text"
          placeholder={placeholder}
          value={query}
          onChange={e => search(e.target.value)}
          onFocus={() => results.length > 0 && setOpen(true)}
        />
        {busy && <span className="loc-spin" />}

        {/* Dropper button */}
        <button
          className={`dropper-btn ${isDropping ? "dropper-btn--active" : ""}`}
          onClick={onDropper}
          title={isDropping ? "Cancel map pick" : "Pick location from map"}
        >
          <svg viewBox="0 0 24 24" xmlns="http://www.w3.org/2000/svg">
            <path d="M12 2C8.13 2 5 5.13 5 9c0 5.25 7 13 7 13s7-7.75 7-13c0-3.87-3.13-7-7-7z" fill="currentColor"/>
            <circle cx="12" cy="9" r="2.5" fill="white"/>
          </svg>
        </button>
      </div>

      {open && (
        <ul className="loc-dropdown">
          {results.map(r => (
            <li key={r.place_id} className="loc-option" onMouseDown={() => pick(r)}>
              <span className="loc-icon">??</span>
              <span className="loc-name">{r.display_name}</span>
            </li>
          ))}
        </ul>
      )}
    </div>
  );
}

/* -- Severity config ------------------------------------------- */

const SEV: Record<number, { label: string; color: string; bg: string; emoji: string }> = {
  1: { label: "Low Risk",      color: "#16a34a", bg: "#dcfce7", emoji: "?" },
  2: { label: "Moderate Risk", color: "#d97706", bg: "#fef3c7", emoji: "??" },
  3: { label: "High Risk",     color: "#ea580c", bg: "#ffedd5", emoji: "??" },
  4: { label: "Severe Risk",   color: "#dc2626", bg: "#fee2e2", emoji: "??" },
};
const sevMeta = (n: number) => SEV[n] ?? { label: "Unknown", color: "#6b7280", bg: "#f3f4f6", emoji: "?" };

/* -- Haversine ------------------------------------------------- */

function haversineMi(a: Location, b: Location) {
  const R = 3958.8;
  const dLat = ((b.lat - a.lat) * Math.PI) / 180;
  const dLng = ((b.lng - a.lng) * Math.PI) / 180;
  const x = Math.sin(dLat / 2) ** 2 +
    Math.cos((a.lat * Math.PI) / 180) * Math.cos((b.lat * Math.PI) / 180) * Math.sin(dLng / 2) ** 2;
  return R * 2 * Math.atan2(Math.sqrt(x), Math.sqrt(1 - x));
}

function sampleRoutePoints(
  coordinates: [number, number][],
  intervalKm = 1
): [number, number][] {
  if (coordinates.length === 0) {
    return [];
  }

  const sampled: [number, number][] = [coordinates[0]];

  let distanceSinceLastSample = 0;

  for (let i = 1; i < coordinates.length; i++) {
    const previous: Location = {
      lat: coordinates[i - 1][0],
      lng: coordinates[i - 1][1],
    };

    const current: Location = {
      lat: coordinates[i][0],
      lng: coordinates[i][1],
    };

    const segmentDistance =
      haversineMi(previous, current) * 1.60934;

    distanceSinceLastSample += segmentDistance;

    if (distanceSinceLastSample >= intervalKm) {
      sampled.push(coordinates[i]);
      distanceSinceLastSample = 0;
    }
  }

  // Always include destination
  const last = coordinates[coordinates.length - 1];

  if (sampled[sampled.length - 1] !== last) {
    sampled.push(last);
  }

  return sampled;
}

/* -- Map click handler ----------------------------------------- */

function MapClickHandler({ pickTarget, onPick }: { pickTarget: PickTarget; onPick: (l: Location) => void }) {
  useMapEvents({
    click(e) {
      if (pickTarget !== null) onPick({ lat: e.latlng.lat, lng: e.latlng.lng });
    },
  });
  return null;
}

/* -- App ------------------------------------------------------- */

export default function App() {
  const [start,      setStart]      = useState<Location | null>(null);
  const [dest,       setDest]       = useState<Location | null>(null);
  const [pickTarget, setPickTarget] = useState<PickTarget>("start");
  const [result,     setResult]     = useState<PredictionResponse | null>(null);
  const [route, setRoute] = useState<RouteResponse | null>(null);
  const [routeLoading, setRouteLoading] = useState(false);
  const [error,      setError]      = useState("");
  const [weather,    setWeather]    = useState("Clear");
  const [hour,       setHour]       = useState(new Date().getHours());
  const [analyzing,  setAnalyzing]  = useState(false);
  const resultRef = useRef<HTMLDivElement>(null);

  /* Auto-scroll result card into view */
  useEffect(() => {
    if (result && resultRef.current) {
      resultRef.current.scrollIntoView({ behavior: 'smooth', block: 'start' });
    }
  }, [result]);

  const isReady   = start !== null && dest !== null;
  const isPicking = pickTarget !== null;

  /* -- Map click -------------------------------------------- */
  const handleMapPick = useCallback((loc: Location) => {
    if (pickTarget === "start") { setStart(loc); setPickTarget("dest"); setResult(null); }
    else if (pickTarget === "dest") { setDest(loc); setPickTarget(null); setResult(null); }
  }, [pickTarget]);

  /* -- Dropper toggle --------------------------------------- */
  const toggleDropper = (target: "start" | "dest") =>
    setPickTarget(prev => prev === target ? null : target);

  /* -- Quick city preset pick ------------------------------- */
  const handleCityPick = (city: Location) => {
    const target: PickTarget = pickTarget ?? (!start ? "start" : !dest ? "dest" : null);
    if (target === "start") {
      setStart(city);
      setPickTarget(!dest ? "dest" : null);
      setResult(null);
    } else if (target === "dest") {
      setDest(city);
      setPickTarget(null);
      setResult(null);
    }
  };

  /* -- Reset ------------------------------------------------ */
  const reset = () => {
    setStart(null); setDest(null);
    setPickTarget("start"); setResult(null); setRoute(null); setError("");
  };

  const getRoute = async (
    from: Location,
    to: Location
  ): Promise<RouteResponse> => {
    setRouteLoading(true);

    try {
      const url =
        `https://router.project-osrm.org/route/v1/driving/` +
        `${from.lng},${from.lat};${to.lng},${to.lat}` +
        `?overview=full&geometries=geojson`;

      const res = await fetch(url);

      if (!res.ok) {
        throw new Error("Could not calculate road route.");
      }

      const data = await res.json();

      if (!data.routes || data.routes.length === 0) {
        throw new Error("No road route found.");
      }

      const selectedRoute = data.routes[0];

      const coordinates: [number, number][] =
        selectedRoute.geometry.coordinates.map(
          ([lng, lat]: [number, number]) => [lat, lng]
        );

      const routeData: RouteResponse = {
        coordinates,
        distance: selectedRoute.distance / 1609.344,
        duration: selectedRoute.duration / 60,
      };

      setRoute(routeData);

      return routeData;

    } catch (e: unknown) {
      setRoute(null);
      throw e instanceof Error
        ? e
        : new Error("Could not calculate road route.");
    } finally {
      setRouteLoading(false);
    }
  };

  /* -- Analyse ---------------------------------------------- */
  const predict = async () => {
    if (!start || !dest) return;
    setAnalyzing(true);
    setError("");
    setResult(null);

    let currentRoute: RouteResponse;

    try {
      currentRoute = await getRoute(start, dest);
    } catch (e: unknown) {
      setError(
        e instanceof Error
          ? e.message
          : "Could not calculate road route."
      );
      setAnalyzing(false);
      return;
    }

    const routePoints = sampleRoutePoints(
      currentRoute.coordinates,
      1
    );

    console.log("Total route points:", currentRoute.coordinates.length);
    console.log("Sampled route points:", routePoints.length);
    console.log("Route points:", routePoints);

    const now   = new Date();
    const month = now.getMonth() + 1;
    const dow   = now.getDay();
    const hasRain = weather === "Rain" || weather === "Thunderstorm";
    const isMorn  = hour >= 7  && hour <= 10 ? 1 : 0;
    const isEve   = hour >= 17 && hour <= 20 ? 1 : 0;

    const body = {
      Distance_mi: parseFloat(
        currentRoute.distance.toFixed(3)
      ),
      Year:                   now.getFullYear(),
      Start_Lng:              start.lng,
      Start_Lat:              start.lat,
      Pressure_in:            29.92,
      Temperature_F:          75,
      Month:                  month,
      Humidity_percent:       hasRain ? 85 : 55,
      Hour:                   hour,
      Wind_Speed_mph:         hasRain ? 12 : 6,
      Quarter:                Math.ceil(month / 3),
      DayOfWeek:              dow,
      Traffic_Signal:         1,
      Weather_Category:       weather,
      Visibility_mi:          weather === "Fog" ? 0.5 : weather === "Snow" ? 1.5 : 8,
      NearRoadInfrastructure: 1,
      Crossing:               0,
      Junction:               1,
      IsWeekend:              dow === 0 || dow === 6 ? 1 : 0,
      IsNight:                hour < 6 || hour >= 20 ? 1 : 0,
      IsRushHour:             isMorn || isEve,
      Precipitation_in:       hasRain ? 0.25 : 0,
      MorningRushHour:        isMorn,
      EveningRushHour:        isEve,
      Stop:                   1,
      HasPrecipitation:       hasRain ? 1 : 0,
      LowVisibility:          weather === "Fog" || weather === "Snow" ? 1 : 0,
      Railway:                0,
    };

    const ctrl  = new AbortController();
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
    } catch (e: unknown) {
      clearTimeout(timer);
      setError(e instanceof DOMException && e.name === "AbortError"
        ? "Request timed out. Try again."
        : e instanceof Error ? e.message : "Could not reach backend.");
    } finally { setAnalyzing(false); }
  };

  /* -- Derived UI ------------------------------------------- */
  const hint =
    analyzing              ? "?? Analysing route safety�"              :
    result                 ? "? Analysis complete"                      :
    pickTarget === "start" ? "?? Click map or search to set start"      :
    pickTarget === "dest"  ? "?? Click map or search to set destination" :
                             "? Conditions set � ready to analyse";

  const dotClass =
    analyzing              ? "step-dot--loading"    :
    result                 ? "step-dot--result"     :
    pickTarget === "start" ? "step-dot--pick-start" :
    pickTarget === "dest"  ? "step-dot--pick-dest"  : "step-dot--ready";

  const meta      = result ? sevMeta(result.severity) : null;
  const polyColor = meta ? meta.color : "#1a73e8";

  return (
    <div className="app">

      {/* -- Sidebar ------------------------------------ */}
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
        <div className={`step-pill ${analyzing ? "step-pill--pulse" : ""}`}>
          <span className={`step-dot ${dotClass}`} />
          <span>{hint}</span>
        </div>

        {/* Pin rows */}
        <div className="pins-block">
          <div className={`pin-row ${start ? "pin-row--active" : ""}`}>
            <div className="pin-circle pin-circle--start">A</div>
            <div className="pin-detail">
              <span>Start point</span>
              <LocationSearch
                placeholder="Search start location�"
                value={start}
                onSelect={loc => { setStart(loc); setPickTarget(!dest ? "dest" : null); setResult(null); }}
                isDropping={pickTarget === "start"}
                onDropper={() => toggleDropper("start")}
              />
            </div>
            {start && (
              <button className="pin-x" onClick={() => { setStart(null); setPickTarget("start"); setResult(null); }} title="Remove">?</button>
            )}
          </div>

          <div className="pin-line" />

          <div className={`pin-row ${dest ? "pin-row--active" : ""}`}>
            <div className="pin-circle pin-circle--dest">B</div>
            <div className="pin-detail">
              <span>Destination</span>
              <LocationSearch
                placeholder="Search destination�"
                value={dest}
                onSelect={loc => { setDest(loc); setPickTarget(null); setResult(null); }}
                isDropping={pickTarget === "dest"}
                onDropper={() => toggleDropper("dest")}
              />
            </div>
            {dest && (
              <button className="pin-x" onClick={() => { setDest(null); setPickTarget("dest"); setResult(null); }} title="Remove">?</button>
            )}
          </div>
        </div>

        {/* -- Quick city presets ------------------------- */}
        <div className="quick-cities">
          <div className="quick-cities-header">
            <p className="cond-title" style={{ margin: 0 }}>Quick locations</p>
            {pickTarget && (
              <span className={`setting-badge setting-badge--${pickTarget}`}>
                Setting: {pickTarget === "start" ? "? Start" : "? Dest"}
              </span>
            )}
            {!pickTarget && isReady && (
              <span className="setting-badge setting-badge--done">Both set ?</span>
            )}
          </div>
          <div className="city-chips">
            {QUICK_CITIES.map(city => {
              const isStart = start?.label === city.label;
              const isDest  = dest?.label  === city.label;
              return (
                <button
                  key={city.label}
                  className={`city-chip ${isStart ? "city-chip--start" : isDest ? "city-chip--dest" : ""}`}
                  onClick={() => handleCityPick(city)}
                  title={isStart ? "Currently: Start" : isDest ? "Currently: Destination" : "Click to set"}
                >
                  {isStart && <span className="city-badge city-badge--a">A</span>}
                  {isDest  && <span className="city-badge city-badge--b">B</span>}
                  {city.label}
                </button>
              );
            })}
          </div>
        </div>

        {/* Conditions */}
        <div className="conditions">
          <p className="cond-title">Weather condition</p>
          <div className="weather-chips">
            {[
              ["Clear","??"], ["Cloudy","??"], ["Rain","???"],
              ["Fog","???"],  ["Snow","??"],  ["Thunderstorm","??"],
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
            Departure time � <strong>{String(hour).padStart(2, "0")}:00</strong>
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
            disabled={!isReady || analyzing}
            onClick={predict}
          >
            {analyzing ? <span className="spin" /> : "Analyse Route Safety"}
          </button>
          {(start || dest) && (
            <button className="btn-ghost" onClick={reset}>Reset map</button>
          )}
        </div>

        {/* Error */}
        {error && <div className="err-box">{error}</div>}

        {/* Result */}
        {result && meta && (
          <div ref={resultRef} className="result-card" style={{ "--c": meta.color, "--bg": meta.bg } as React.CSSProperties}>

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

            {route && (
              <div className="result-meta">
                <div className="meta-chip">
                  <span>📍</span>
                  <strong>{route.distance.toFixed(2)} mi</strong>
                </div>

                <div className="meta-chip">
                  <span>⏱</span>
                  <strong>{Math.round(route.duration)} min</strong>
                </div>

                <div className="meta-chip">
                  <span>🌦</span>
                  <strong>{weather}</strong>
                </div>

                <div className="meta-chip">
                  <span>🕐</span>
                  <strong>{hour}:00</strong>
                </div>
              </div>
            )}
          </div>
        )}

      </aside>

      {/* -- Map ------------------------------------------ */}
      <div className={`map-area ${isPicking ? "map-area--picking" : ""}`}>
        <MapContainer center={[20.59, 78.96]} zoom={5} className="lmap" zoomControl={false}>
          <TileLayer
            attribution='&copy; <a href="https://www.openstreetmap.org/copyright">OpenStreetMap</a>'
            url="https://{s}.tile.openstreetmap.org/{z}/{x}/{y}.png"
          />

          <MapClickHandler pickTarget={pickTarget} onPick={handleMapPick} />

          {start && (
            <Marker position={[start.lat, start.lng]} icon={startPin}>
              <Popup><strong>📍 Start</strong><br />{start.label ?? `${start.lat.toFixed(5)}, ${start.lng.toFixed(5)}`}</Popup>
            </Marker>
          )}

          {dest && (
            <Marker position={[dest.lat, dest.lng]} icon={destPin}>
              <Popup><strong>🏁 Destination</strong><br />{dest.label ?? `${dest.lat.toFixed(5)}, ${dest.lng.toFixed(5)}`}</Popup>
            </Marker>
          )}

          {route && (
            <Polyline
              positions={route.coordinates}
              pathOptions={{
                color: polyColor,
                weight: 6,
                opacity: 0.9,
              }}
            />
          )}
        </MapContainer>

        {isPicking && (
          <div className="map-hint">
            {pickTarget === "start" ? "?? Click to set start" : "?? Click to set destination"}
          </div>
        )}
      </div>

    </div>
  );
}
