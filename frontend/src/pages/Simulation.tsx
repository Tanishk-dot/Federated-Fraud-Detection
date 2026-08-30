import { AnimatePresence, motion } from "framer-motion"
import { useEffect, useRef, useState } from "react"
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts"
import { api, type SimEvent, type SimJobStatus } from "../api"
import { Badge, Button, Card, SectionTitle } from "../components/ui"
import { EASE, Reveal } from "../components/motion"

const NUM_CLIENTS = 10
const SIZE = 460
const CENTER = SIZE / 2
const RADIUS = 178
const NODE = 60
const SERVER_NODE = 84

function clientPos(i: number) {
  const angle = (i / NUM_CLIENTS) * 2 * Math.PI - Math.PI / 2
  return { x: CENTER + RADIUS * Math.cos(angle), y: CENTER + RADIUS * Math.sin(angle) }
}
const SERVER_POS = { x: CENTER, y: CENTER }

type Activity = "idle" | "selected" | "training" | "sent"
type Flash = "correct" | "missed" | "false-alarm" | null

interface Packet { id: string; from: { x: number; y: number }; to: { x: number; y: number }; color: string }
interface RoundMetric { round: number; accuracy: number; f1: number; recall: number; roc_auc: number }

function describeEvent(e: SimEvent): string {
  switch (e.type) {
    case "warm_start":
      return e.checkpoint
        ? `Loaded the real trained checkpoint (${e.checkpoint}) as a starting point`
        : "No checkpoint found — starting from a fresh model"
    case "data_ready":
      return `Generated ${e.clients} banks × ${e.samples_per_client} transactions each (${((e.fraud_rate ?? 0) * 100).toFixed(1)}% fraud)`
    case "dp_config":
      return `Differential privacy on: ε=${e.epsilon}, δ=${e.delta}, clip norm ${e.clip_norm}`
    case "round_start":
      return `— Round ${e.round}/${e.total_rounds} —`
    case "clients_selected":
      return `Server selected banks ${(e.clients as number[]).map((c) => c + 1).join(", ")} for this round`
    case "client_training":
      return `Bank ${(e.client ?? 0) + 1} training locally on its own data (never leaves the bank)`
    case "client_done":
      return `Bank ${(e.client ?? 0) + 1} sends its clipped + noised update to the server`
    case "server_aggregating":
      return "Server aggregating updates (FedProx)"
    case "server_aggregated":
      return "Server broadcasts the updated global model to every bank"
    case "round_retry":
      return `Round ${e.round}: that update looked unstable — retrying (attempt ${e.attempt})`
    case "round_complete":
      return `Round ${e.round} done — accuracy ${((e.accuracy ?? 0) * 100).toFixed(1)}%, F1 ${(e.f1 ?? 0).toFixed(2)}, recall ${((e.recall ?? 0) * 100).toFixed(0)}%`
    case "transaction_scored":
      return `Bank ${(e.client ?? 0) + 1}: $${e.amount} at ${e.merchant_type} → ${e.decision}` +
        (e.actual === "FRAUD" ? (e.correct ? " ✓ real fraud caught" : " ✗ missed") : (e.correct ? "" : " ✗ false alarm"))
    case "simulation_done":
      return "Simulation complete."
    default:
      return e.type
  }
}

function Node({
  x, y, size, label, sub, active, flash, pulsing, accent = "#3987e5",
}: {
  x: number; y: number; size: number; label: string; sub?: string
  active?: boolean; flash?: Flash; pulsing?: boolean; accent?: string
}) {
  const flashColor = flash === "correct" ? "#0ca30c" : flash === "missed" ? "#d03b3b" : flash === "false-alarm" ? "#fab219" : null
  return (
    <div
      className="absolute flex flex-col items-center justify-center rounded-full border text-center transition-colors duration-300"
      style={{
        left: x - size / 2, top: y - size / 2, width: size, height: size,
        borderColor: flashColor ?? (active ? accent : "rgba(255,255,255,0.14)"),
        background: flashColor ? `${flashColor}22` : active ? `${accent}1a` : "var(--color-surface-2)",
        boxShadow: pulsing ? `0 0 0 6px ${accent}22` : flashColor ? `0 0 0 6px ${flashColor}22` : "none",
      }}
    >
      <span className="text-[11px] font-bold text-ink-primary leading-none">{label}</span>
      {sub && <span className="text-[9px] text-ink-muted mt-0.5 leading-none">{sub}</span>}
      {pulsing && (
        <motion.span
          className="absolute inset-0 rounded-full border"
          style={{ borderColor: accent }}
          initial={{ opacity: 0.6, scale: 1 }}
          animate={{ opacity: 0, scale: 1.6 }}
          transition={{ duration: 1, repeat: Infinity, ease: "easeOut" }}
        />
      )}
    </div>
  )
}

export default function Simulation() {
  const [jobId, setJobId] = useState<string | null>(null)
  const [backendStatus, setBackendStatus] = useState<SimJobStatus["status"] | "idle">("idle")
  const [backendError, setBackendError] = useState<string | null>(null)
  const [allEvents, setAllEvents] = useState<SimEvent[]>([])

  const [activity, setActivity] = useState<Record<number, Activity>>({})
  const [flash, setFlash] = useState<Record<number, Flash>>({})
  const [serverPulsing, setServerPulsing] = useState(false)
  const [packets, setPackets] = useState<Packet[]>([])
  const [roundInfo, setRoundInfo] = useState<{ round: number; total: number } | null>(null)
  const [dataInfo, setDataInfo] = useState<{ clients: number; samples: number; fraudRate: number } | null>(null)
  const [dpInfo, setDpInfo] = useState<{ epsilon: number; delta: number } | null>(null)
  const [warmStart, setWarmStart] = useState<{ checkpoint: string | null } | null>(null)
  const [metricsHistory, setMetricsHistory] = useState<RoundMetric[]>([])
  const [feed, setFeed] = useState<(SimEvent & { id: string })[]>([])
  const [log, setLog] = useState<string[]>([])
  const [playbackDone, setPlaybackDone] = useState(false)

  const allEventsRef = useRef<SimEvent[]>([])
  const playIdxRef = useRef(0)
  const logRef = useRef<HTMLDivElement>(null)

  useEffect(() => { allEventsRef.current = allEvents }, [allEvents])

  const spawnPacket = (from: { x: number; y: number }, to: { x: number; y: number }, color: string) => {
    const id = `${Date.now()}-${Math.random()}`
    setPackets((p) => [...p, { id, from, to, color }])
    window.setTimeout(() => setPackets((p) => p.filter((x) => x.id !== id)), 750)
  }

  const flashClient = (cid: number, kind: Flash) => {
    setFlash((prev) => ({ ...prev, [cid]: kind }))
    window.setTimeout(() => setFlash((prev) => ({ ...prev, [cid]: null })), 1000)
  }

  const processEvent = (e: SimEvent) => {
    setLog((prev) => [...prev.slice(-49), describeEvent(e)])
    switch (e.type) {
      case "warm_start":
        setWarmStart({ checkpoint: e.checkpoint ?? null })
        break
      case "data_ready":
        setDataInfo({ clients: (e.clients as number) ?? NUM_CLIENTS, samples: e.samples_per_client ?? 0, fraudRate: e.fraud_rate ?? 0 })
        break
      case "dp_config":
        setDpInfo({ epsilon: e.epsilon ?? 0, delta: e.delta ?? 0 })
        break
      case "round_start":
        setRoundInfo({ round: e.round ?? 0, total: e.total_rounds ?? 0 })
        setActivity({})
        break
      case "clients_selected": {
        const clients = (e.clients as number[]) ?? []
        setActivity((prev) => {
          const next = { ...prev }
          clients.forEach((c) => { next[c] = "selected" })
          return next
        })
        clients.forEach((c) => spawnPacket(clientPos(c), SERVER_POS, "#3987e5"))
        break
      }
      case "client_training":
        setActivity((prev) => ({ ...prev, [e.client ?? 0]: "training" }))
        break
      case "client_done":
        setActivity((prev) => ({ ...prev, [e.client ?? 0]: "sent" }))
        break
      case "server_aggregating":
        setServerPulsing(true)
        break
      case "server_aggregated":
        setServerPulsing(false)
        for (let c = 0; c < NUM_CLIENTS; c++) spawnPacket(SERVER_POS, clientPos(c), "#9085e9")
        break
      case "round_complete":
        setMetricsHistory((prev) => [...prev, {
          round: e.round ?? 0, accuracy: e.accuracy ?? 0, f1: e.f1 ?? 0,
          recall: e.recall ?? 0, roc_auc: e.roc_auc ?? 0,
        }])
        break
      case "transaction_scored": {
        const cid = e.client ?? 0
        const kind: Flash = e.correct ? "correct" : e.actual === "FRAUD" ? "missed" : "false-alarm"
        flashClient(cid, kind)
        setFeed((prev) => [{ ...e, id: `${Date.now()}-${Math.random()}` }, ...prev].slice(0, 30))
        break
      }
      case "simulation_done":
        setPlaybackDone(true)
        break
    }
  }

  // Single persistent playback clock - reads from refs so it never needs to
  // restart when new events arrive from polling; it just idles until the
  // next unplayed event exists. This paces the animation for legibility
  // (one event every ~260ms) independent of how fast the backend actually
  // computed each step.
  useEffect(() => {
    const id = window.setInterval(() => {
      const events = allEventsRef.current
      if (playIdxRef.current < events.length) {
        processEvent(events[playIdxRef.current])
        playIdxRef.current += 1
      }
    }, 260)
    return () => window.clearInterval(id)
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [])

  useEffect(() => {
    if (!jobId) return
    let cancelled = false
    let timer: number | undefined
    const poll = async () => {
      try {
        const s = await api.simStatus(jobId)
        if (cancelled) return
        setAllEvents(s.events)
        setBackendStatus(s.status)
        setBackendError(s.error)
        if (s.status === "running" || s.status === "starting") {
          timer = window.setTimeout(poll, 900)
        }
      } catch (err) {
        if (!cancelled) setBackendError(String(err))
      }
    }
    poll()
    return () => { cancelled = true; if (timer) window.clearTimeout(timer) }
  }, [jobId])

  useEffect(() => {
    logRef.current?.scrollTo({ top: logRef.current.scrollHeight })
  }, [log])

  const reset = () => {
    setJobId(null); setBackendStatus("idle"); setBackendError(null); setAllEvents([])
    allEventsRef.current = []; playIdxRef.current = 0
    setActivity({}); setFlash({}); setServerPulsing(false); setPackets([])
    setRoundInfo(null); setDataInfo(null); setDpInfo(null); setWarmStart(null)
    setMetricsHistory([]); setFeed([]); setLog([]); setPlaybackDone(false)
  }

  const run = async () => {
    reset()
    try {
      const { job_id } = await api.simStart({})
      setJobId(job_id)
      setBackendStatus("starting")
    } catch (e) {
      setBackendError(String(e))
    }
  }

  const isActive = jobId !== null && !(playbackDone && backendStatus === "complete")
  const latest = metricsHistory[metricsHistory.length - 1]

  return (
    <div>
      <SectionTitle
        eyebrow="Watch it work"
        title="Live Simulation"
        sub="Generates a fresh synthetic dataset for 10 banks, then runs the real FedProx + client-level DP-FedAvg loop round by round — same model, same mechanism as everywhere else in this app, just small and fast enough to watch happen."
      />

      <Reveal className="mb-6 flex flex-wrap items-center gap-3">
        <Button onClick={run} disabled={isActive}>
          {isActive ? "Running…" : jobId ? "Run again" : "▶ Run simulation"}
        </Button>
        {roundInfo && (
          <Badge status="warning">Round {roundInfo.round} / {roundInfo.total}</Badge>
        )}
        {warmStart && (
          <Badge status={warmStart.checkpoint ? "good" : "neutral"}>
            {warmStart.checkpoint ? `warm-started from ${warmStart.checkpoint}` : "fresh model"}
          </Badge>
        )}
        {dpInfo && <Badge status="good">DP: ε={dpInfo.epsilon}, δ={dpInfo.delta}</Badge>}
        {playbackDone && backendStatus === "complete" && <Badge status="good">✅ complete</Badge>}
        {backendError && <Badge status="critical">error: {backendError}</Badge>}
      </Reveal>

      <Card className="mb-8">
        <div className="flex flex-col lg:flex-row gap-8 items-center">
          <div className="relative shrink-0" style={{ width: SIZE, height: SIZE }}>
            {/* connecting spokes */}
            <svg className="absolute inset-0 pointer-events-none" width={SIZE} height={SIZE}>
              {Array.from({ length: NUM_CLIENTS }).map((_, i) => {
                const p = clientPos(i)
                return (
                  <line key={i} x1={SERVER_POS.x} y1={SERVER_POS.y} x2={p.x} y2={p.y}
                    stroke="rgba(255,255,255,0.06)" strokeWidth={1} />
                )
              })}
            </svg>

            <AnimatePresence>
              {packets.map((p) => (
                <motion.div
                  key={p.id}
                  className="absolute h-2.5 w-2.5 rounded-full z-10"
                  style={{ background: p.color, boxShadow: `0 0 8px ${p.color}` }}
                  initial={{ left: p.from.x - 5, top: p.from.y - 5, opacity: 1 }}
                  animate={{ left: p.to.x - 5, top: p.to.y - 5, opacity: [1, 1, 0] }}
                  exit={{ opacity: 0 }}
                  transition={{ duration: 0.7, ease: EASE }}
                />
              ))}
            </AnimatePresence>

            {Array.from({ length: NUM_CLIENTS }).map((_, i) => {
              const p = clientPos(i)
              const act = activity[i] ?? "idle"
              return (
                <Node
                  key={i} x={p.x} y={p.y} size={NODE}
                  label={`Bank ${i + 1}`}
                  sub={act === "training" ? "training…" : act === "sent" ? "sent" : act === "selected" ? "selected" : undefined}
                  active={act !== "idle"} flash={flash[i] ?? null}
                />
              )
            })}

            <Node
              x={SERVER_POS.x} y={SERVER_POS.y} size={SERVER_NODE}
              label="Server" sub={serverPulsing ? "aggregating" : dataInfo ? "FedProx" : undefined}
              active={serverPulsing} pulsing={serverPulsing} accent="#9085e9"
            />
          </div>

          <div className="flex-1 min-w-0 w-full">
            {!dataInfo && !backendError && (
              <p className="text-ink-muted text-sm">
                Press "Run simulation" — 10 banks, each with their own transactions (some fraud, some
                not), train a shared model round by round without ever sharing raw data. Real training,
                real detections, ~30 seconds.
              </p>
            )}
            {dataInfo && (
              <p className="text-xs text-ink-muted mb-4">
                {dataInfo.clients} banks × {dataInfo.samples} transactions each, {(dataInfo.fraudRate * 100).toFixed(1)}% fraud rate.
              </p>
            )}
            {latest && (
              <div className="grid grid-cols-4 gap-3 mb-4">
                {([
                  ["Accuracy", latest.accuracy], ["F1", latest.f1],
                  ["Recall", latest.recall], ["ROC-AUC", latest.roc_auc],
                ] as const).map(([label, val]) => (
                  <div key={label}>
                    <div className="text-[10px] text-ink-muted uppercase tracking-wide">{label}</div>
                    <div className="text-xl font-extrabold tabular-nums text-series-blue">{(val * 100).toFixed(1)}%</div>
                  </div>
                ))}
              </div>
            )}
            {metricsHistory.length > 1 && (
              <div className="h-40">
                <ResponsiveContainer width="100%" height="100%">
                  <LineChart data={metricsHistory}>
                    <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                    <XAxis dataKey="round" tick={{ fontSize: 10, fill: "#8b8a83" }} axisLine={{ stroke: "rgba(255,255,255,0.1)" }} tickLine={false} />
                    <YAxis domain={[0, 1]} tick={{ fontSize: 10, fill: "#8b8a83" }} axisLine={false} tickLine={false} />
                    <Tooltip contentStyle={{ background: "#1a1a19", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                    <Line type="monotone" dataKey="f1" stroke="#9085e9" strokeWidth={2} dot={false} name="F1" />
                    <Line type="monotone" dataKey="recall" stroke="#199e70" strokeWidth={2} dot={false} name="Recall" />
                    <Line type="monotone" dataKey="roc_auc" stroke="#3987e5" strokeWidth={2} dot={false} name="ROC-AUC" />
                  </LineChart>
                </ResponsiveContainer>
              </div>
            )}
          </div>
        </div>
      </Card>

      <div className="grid md:grid-cols-2 gap-6">
        <Card>
          <h3 className="font-bold text-lg mb-1">Live transaction feed</h3>
          <p className="text-xs text-ink-secondary mb-4">
            One fresh, unseen transaction per bank, scored with the model as it stands after each
            round. Green = caught real fraud, amber = false alarm, red = missed — all three happen
            honestly; this isn't tuned to always look perfect.
          </p>
          <div className="max-h-80 overflow-y-auto rounded-xl border border-white/8">
            <table className="w-full text-xs">
              <thead className="sticky top-0 bg-surface-2 text-ink-muted uppercase">
                <tr>
                  <th className="text-left px-3 py-2">Bank</th>
                  <th className="text-left px-3 py-2">Amount</th>
                  <th className="text-left px-3 py-2">Decision</th>
                  <th className="text-left px-3 py-2">Actual</th>
                </tr>
              </thead>
              <tbody>
                {feed.map((e) => (
                  <tr key={e.id} className={`border-t border-white/5 ${e.actual === "FRAUD" ? "bg-status-critical/5" : ""}`}>
                    <td className="px-3 py-2">Bank {(e.client ?? 0) + 1}</td>
                    <td className="px-3 py-2 tabular-nums">${e.amount}</td>
                    <td className="px-3 py-2">
                      <Badge status={e.decision === "FRAUD" ? "critical" : "good"}>{e.decision}</Badge>
                    </td>
                    <td className="px-3 py-2">
                      <Badge status={e.correct ? "good" : "warning"}>
                        {e.actual}{e.correct ? " ✓" : " ✗"}
                      </Badge>
                    </td>
                  </tr>
                ))}
                {feed.length === 0 && (
                  <tr><td colSpan={4} className="px-3 py-6 text-center text-ink-muted">No transactions scored yet.</td></tr>
                )}
              </tbody>
            </table>
          </div>
        </Card>

        <Card>
          <h3 className="font-bold text-lg mb-1">What's happening</h3>
          <p className="text-xs text-ink-secondary mb-4">
            A plain-language log of the real event stream driving the animation above.
          </p>
          <div ref={logRef} className="max-h-80 overflow-y-auto rounded-xl border border-white/8 bg-surface-1 p-3 font-mono text-[11px] leading-relaxed text-ink-secondary">
            {log.length === 0 && <div className="text-ink-muted">Waiting to start…</div>}
            {log.map((line, i) => <div key={i}>{line}</div>)}
          </div>
        </Card>
      </div>
    </div>
  )
}
