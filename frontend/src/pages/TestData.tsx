import { AnimatePresence, motion } from "framer-motion"
import { type ReactNode, useEffect, useMemo, useRef, useState } from "react"
import {
  Bar, BarChart, CartesianGrid, Cell, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts"
import {
  api, type CsvInspectResponse, type CsvPredictResponse, type ModelInfo, type Transaction,
} from "../api"
import { Badge, Button, Card, ProgressBar, SectionTitle } from "../components/ui"
import { EASE, Reveal } from "../components/motion"

function DecisionBadge({ decision }: { decision: "FRAUD" | "LEGITIMATE" }) {
  return decision === "FRAUD" ? (
    <Badge status="critical">🚨 fraud</Badge>
  ) : (
    <Badge status="good">✅ legitimate</Badge>
  )
}

// ---------------------------------------------------------------------------
// Single transaction
// ---------------------------------------------------------------------------

function SingleTransaction({ info }: { info: ModelInfo | null }) {
  const [txn, setTxn] = useState<Transaction>({
    amount: 45.5, hour: 14, day_of_week: 2, merchant_type: "Grocery",
    distance_km: 2, minutes_since_last: 180, card_age_days: 730, txns_last_24h: 2,
  })
  const [result, setResult] = useState<Awaited<ReturnType<typeof api.predict>> | null>(null)
  const [loading, setLoading] = useState(false)
  const [error, setError] = useState<string | null>(null)

  const run = async () => {
    setLoading(true); setError(null)
    try {
      setResult(await api.predict(txn))
    } catch (e) {
      setError(String(e))
    } finally {
      setLoading(false)
    }
  }

  const field = (key: keyof Transaction, label: string, step = 1) => (
    <label className="flex flex-col gap-1 text-sm">
      <span className="text-ink-muted">{label}</span>
      <input
        type="number" step={step} value={txn[key] as number}
        onChange={(e) => setTxn({ ...txn, [key]: parseFloat(e.target.value) || 0 })}
        className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-ink-primary focus:outline-none focus:border-series-blue"
      />
    </label>
  )

  return (
    <Reveal><Card>
      <h3 className="font-bold text-lg mb-4">Try a single transaction</h3>
      <div className="grid grid-cols-2 md:grid-cols-4 gap-4 mb-4">
        {field("amount", "Amount ($)")}
        {field("hour", "Hour (0-23)")}
        {field("day_of_week", "Day of week (0=Mon)")}
        <label className="flex flex-col gap-1 text-sm">
          <span className="text-ink-muted">Merchant type</span>
          <input
            list="merchant-type-options"
            value={txn.merchant_type}
            onChange={(e) => setTxn({ ...txn, merchant_type: e.target.value })}
            placeholder="e.g. Online, or type your own"
            className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-ink-primary focus:outline-none focus:border-series-blue"
          />
          <datalist id="merchant-type-options">
            {(info?.merchant_types ?? [
              "Grocery", "Gas Station", "Pharmacy", "Restaurant", "ATM",
              "Retail", "Electronics", "Online", "Travel", "Jewelry",
            ]).map((m) => <option key={m} value={m} />)}
          </datalist>
          <span className="text-[10px] text-ink-muted">
            Not in the list? Type anything — unrecognized types score as medium risk.
          </span>
        </label>
        {field("distance_km", "Distance from last (km)")}
        {field("minutes_since_last", "Minutes since last txn")}
        {field("card_age_days", "Card age (days)")}
        {field("txns_last_24h", "Transactions in last 24h")}
      </div>

      <Button onClick={run} disabled={loading}>{loading ? "Running model..." : "Run real prediction"}</Button>
      {error && <p className="text-status-critical text-sm mt-3">{error}</p>}

      {result && (
        <div className="mt-6 flex items-center gap-6 flex-wrap">
          <DecisionBadge decision={result.decision} />
          <div>
            <div className="text-xs text-ink-muted uppercase">Fraud probability</div>
            <div className="text-2xl font-extrabold tabular-nums text-series-blue">{(result.fraud_prob * 100).toFixed(1)}%</div>
          </div>
          <div>
            <div className="text-xs text-ink-muted uppercase">Anomaly score</div>
            <div className="text-2xl font-extrabold tabular-nums text-series-violet">{(result.anomaly_score * 100).toFixed(1)}%</div>
          </div>
        </div>
      )}
    </Card></Reveal>
  )
}

// ---------------------------------------------------------------------------
// CSV upload with flexible column mapping
// ---------------------------------------------------------------------------

function CsvUpload({ info }: { info: ModelInfo | null }) {
  const [file, setFile] = useState<File | null>(null)
  const [inspect, setInspect] = useState<CsvInspectResponse | null>(null)
  const [mapping, setMapping] = useState<Record<string, string | null>>({})
  const [accountIdCol, setAccountIdCol] = useState<string | null>(null)
  const [result, setResult] = useState<CsvPredictResponse | null>(null)
  const [busy, setBusy] = useState(false)
  const [error, setError] = useState<string | null>(null)
  const [liveMode, setLiveMode] = useState(false)
  const [liveIdx, setLiveIdx] = useState(0)
  const liveTimer = useRef<number | null>(null)

  const requiredFields = info?.raw_columns ?? []

  const onFile = async (f: File) => {
    setFile(f); setResult(null); setError(null); setLiveMode(false)
    try {
      const insp = await api.csvInspect(f)
      setInspect(insp)
      setMapping(insp.suggested_mapping)
      setAccountIdCol(insp.suggested_mapping.account_id ?? null)
    } catch (e) {
      setError(String(e))
    }
  }

  const runPredict = async () => {
    if (!file) return
    setBusy(true); setError(null)
    try {
      const res = await api.csvPredict(file, mapping, accountIdCol)
      setResult(res)
    } catch (e) {
      setError(String(e))
    } finally {
      setBusy(false)
    }
  }

  // Live feed: reveal one result at a time.
  useEffect(() => {
    if (!liveMode || !result) return
    if (liveIdx >= result.results.length) return
    liveTimer.current = window.setTimeout(() => setLiveIdx((i) => i + 1), 350)
    return () => { if (liveTimer.current) clearTimeout(liveTimer.current) }
  }, [liveMode, liveIdx, result])

  const visibleResults = liveMode ? (result?.results ?? []).slice(0, liveIdx) : result?.results ?? []
  const histBuckets = useMemo(() => {
    const buckets = Array.from({ length: 10 }, (_, i) => ({ range: `${i * 10}-${i * 10 + 10}%`, count: 0 }))
    for (const r of result?.results ?? []) {
      const bucket = Math.min(9, Math.floor(r.fraud_prob * 10))
      buckets[bucket].count++
    }
    return buckets
  }, [result])

  return (
    <Reveal><Card>
      <h3 className="font-bold text-lg mb-1">Upload any transaction CSV</h3>
      <p className="text-sm text-ink-secondary mb-4">
        Any schema works — after upload, map your columns to what the model needs.
        Unmapped fields fall back to a neutral default (shown per-field below), so even a
        CSV missing most fields (e.g. a raw PaySim export) still runs.
      </p>

      <input
        type="file" accept=".csv"
        onChange={(e) => e.target.files?.[0] && onFile(e.target.files[0])}
        className="text-sm file:mr-4 file:rounded-lg file:border-0 file:bg-series-blue/20 file:text-series-blue file:px-4 file:py-2 file:font-semibold file:cursor-pointer"
      />
      {error && <p className="text-status-critical text-sm mt-3">{error}</p>}

      {inspect && (
        <div className="mt-6">
          <div className="text-sm text-ink-muted mb-3">
            {inspect.num_rows.toLocaleString()} rows detected · {inspect.columns.length} columns
            {inspect.num_rows > 3000 && (
              <span className="text-status-warning ml-2">
                — this is an interactive test tool, not a batch-scoring pipeline; it'll run on a
                random sample of up to 3,000 rows, not the full file
              </span>
            )}
          </div>

          <div className="grid md:grid-cols-2 gap-3 mb-4">
            {requiredFields.map((field) => (
              <label key={field} className="flex flex-col gap-1 text-sm">
                <span className="text-ink-muted">
                  {field}
                  {!mapping[field] && (
                    <span className="text-status-warning ml-2">
                      (unmapped → default: {String(info?.defaults[field])})
                    </span>
                  )}
                </span>
                <select
                  value={mapping[field] ?? ""}
                  onChange={(e) => setMapping({ ...mapping, [field]: e.target.value || null })}
                  className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-ink-primary focus:outline-none focus:border-series-blue"
                >
                  <option value="">— use default —</option>
                  {inspect.columns.map((c) => <option key={c} value={c}>{c}</option>)}
                </select>
              </label>
            ))}
            <label className="flex flex-col gap-1 text-sm">
              <span className="text-ink-muted">account/entity id column (optional — enables real history per account)</span>
              <select
                value={accountIdCol ?? ""}
                onChange={(e) => setAccountIdCol(e.target.value || null)}
                className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 text-ink-primary focus:outline-none focus:border-series-blue"
              >
                <option value="">— none, score each row independently —</option>
                {inspect.columns.map((c) => <option key={c} value={c}>{c}</option>)}
              </select>
            </label>
          </div>

          <div className="flex gap-3">
            <Button onClick={runPredict} disabled={busy}>{busy ? "Running model..." : "Run real predictions"}</Button>
          </div>
        </div>
      )}

      {result && (
        <div className="mt-8">
          {result.truncated && result.truncation_note && (
            <div className="mb-4 rounded-lg border border-status-warning/30 bg-status-warning/10 px-3 py-2 text-xs text-status-warning">
              {result.truncation_note}
            </div>
          )}
          <div className="flex items-center justify-between flex-wrap gap-3 mb-4">
            <div className="flex gap-6">
              <div><span className="text-2xl font-extrabold tabular-nums">{result.num_rows}</span><span className="text-ink-muted text-sm ml-1">rows{result.truncated ? ` (of ${result.total_rows_in_file.toLocaleString()})` : ""}</span></div>
              <div><span className="text-2xl font-extrabold tabular-nums text-status-critical">{result.num_fraud}</span><span className="text-ink-muted text-sm ml-1">flagged fraud</span></div>
              <div><span className="text-2xl font-extrabold tabular-nums text-series-blue">{((result.num_fraud / result.num_rows) * 100).toFixed(1)}%</span><span className="text-ink-muted text-sm ml-1">flag rate</span></div>
            </div>
            <Button
              variant={liveMode ? "danger" : "ghost"}
              onClick={() => { setLiveMode(!liveMode); setLiveIdx(0) }}
            >
              {liveMode ? "■ Stop live simulation" : "▶ Simulate as live feed"}
            </Button>
          </div>

          {!liveMode && (
            <div className="h-48 mb-6">
              <ResponsiveContainer width="100%" height="100%">
                <BarChart data={histBuckets}>
                  <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                  <XAxis dataKey="range" tick={{ fontSize: 11, fill: "#8b8a83" }} axisLine={{ stroke: "rgba(255,255,255,0.1)" }} tickLine={false} />
                  <YAxis tick={{ fontSize: 11, fill: "#8b8a83" }} axisLine={false} tickLine={false} />
                  <Tooltip contentStyle={{ background: "#1a1a19", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                  <Bar dataKey="count" radius={[4, 4, 0, 0]}>
                    {histBuckets.map((_, i) => (
                      <Cell key={i} fill={i >= 7 ? "#e66767" : i >= 4 ? "#c98500" : "#3987e5"} />
                    ))}
                  </Bar>
                </BarChart>
              </ResponsiveContainer>
            </div>
          )}

          {liveMode && (
            <ProgressBar value={liveIdx / result.results.length} />
          )}

          <div className="mt-4 max-h-96 overflow-y-auto rounded-xl border border-white/8">
            <table className="w-full text-sm">
              <thead className="sticky top-0 bg-surface-2 text-ink-muted text-xs uppercase">
                <tr>
                  <th className="text-left px-3 py-2">Row</th>
                  {result.results[0]?.account_id !== undefined && <th className="text-left px-3 py-2">Account</th>}
                  <th className="text-left px-3 py-2">Fraud prob</th>
                  <th className="text-left px-3 py-2">Anomaly</th>
                  <th className="text-left px-3 py-2">Decision</th>
                </tr>
              </thead>
              <tbody>
                {[...visibleResults].reverse().map((r) => (
                  <tr key={r.row} className={`border-t border-white/5 ${r.decision === "FRAUD" ? "bg-status-critical/5" : ""}`}>
                    <td className="px-3 py-2 tabular-nums text-ink-muted">{r.row}</td>
                    {r.account_id !== undefined && <td className="px-3 py-2 font-mono text-xs">{r.account_id}</td>}
                    <td className="px-3 py-2 tabular-nums">{(r.fraud_prob * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2 tabular-nums">{(r.anomaly_score * 100).toFixed(1)}%</td>
                    <td className="px-3 py-2"><DecisionBadge decision={r.decision} /></td>
                  </tr>
                ))}
              </tbody>
            </table>
          </div>
        </div>
      )}
    </Card></Reveal>
  )
}

function TabButton({
  active, onClick, children,
}: { active: boolean; onClick: () => void; children: ReactNode }) {
  return (
    <button
      onClick={onClick}
      className={`relative rounded-lg px-4 py-2 text-sm font-semibold transition-colors ${active ? "text-white" : "text-ink-muted hover:text-white"}`}
    >
      {active && (
        <motion.span
          layoutId="testdata-tab-pill"
          className="absolute inset-0 rounded-lg bg-series-blue/18 border border-series-blue/25"
          transition={{ duration: 0.3, ease: EASE }}
        />
      )}
      <span className="relative z-10">{children}</span>
    </button>
  )
}

export default function TestData() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [mode, setMode] = useState<"single" | "csv">("csv")

  useEffect(() => { api.modelInfo().then(setInfo).catch(() => {}) }, [])

  return (
    <div>
      <SectionTitle
        eyebrow="Real inference"
        title="Test Your Data"
        sub="Runs the actual trained checkpoint, not illustrative rules. Upload any CSV (any schema — map your columns), or try one transaction by hand."
      />

      <div className="flex gap-2 mb-6">
        <TabButton active={mode === "csv"} onClick={() => setMode("csv")}>Upload CSV</TabButton>
        <TabButton active={mode === "single"} onClick={() => setMode("single")}>Single transaction</TabButton>
      </div>

      <AnimatePresence mode="wait">
        <motion.div
          key={mode}
          initial={{ opacity: 0, y: 6 }}
          animate={{ opacity: 1, y: 0 }}
          exit={{ opacity: 0 }}
          transition={{ duration: 0.2, ease: EASE }}
        >
          {mode === "csv" ? <CsvUpload info={info} /> : <SingleTransaction info={info} />}
        </motion.div>
      </AnimatePresence>
    </div>
  )
}
