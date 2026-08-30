import { useEffect, useState } from "react"
import {
  CartesianGrid, Legend, Line, LineChart, ReferenceLine,
  ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts"
import { Badge, Card, SectionTitle } from "../components/ui"
import { Reveal, Stagger, StaggerItem } from "../components/motion"

const LAYERS = [
  { name: "Data locality", guarantee: "Raw data never leaves clients", status: "good", label: "real" },
  { name: "Differential Privacy", guarantee: "Client's whole update is bounded + noised", status: "good", label: "real" },
  { name: "Homomorphic Encryption", guarantee: "Server can't see individual updates", status: "warning", label: "implemented, not wired into training" },
  { name: "Secure Aggregation (proxies)", guarantee: "No single party sees all updates", status: "warning", label: "implemented, not wired into training" },
  { name: "TLS 1.3", guarantee: "Transport security", status: "neutral", label: "n/a — single-process simulation" },
  { name: "Mutual Authentication", guarantee: "Identity verification", status: "neutral", label: "n/a — single-process simulation" },
] as const

interface SweepPoint {
  epsilon: number | null
  accuracy: number
  precision: number
  recall: number
  f1: number
  roc_auc: number
  train_seconds: number
}
interface SweepData {
  dataset: string
  test_set_size: number
  test_fraud_rate: number
  methodology: string
  points: SweepPoint[]
}

const DEFAULT_EPSILON = 100

export default function Architecture() {
  const [sweep, setSweep] = useState<SweepData | null>(null)
  const [sweepError, setSweepError] = useState<string | null>(null)

  useEffect(() => {
    fetch("/api/epsilon-sweep")
      .then((r) => { if (!r.ok) return r.text().then((t) => { throw new Error(t) }); return r.json() })
      .then(setSweep)
      .catch((e) => setSweepError(String(e)))
  }, [])

  // Recharts needs a numeric x-axis; plot "no DP" at a value past the
  // largest swept epsilon so it reads as the ceiling, not epsilon=0.
  const chartData = sweep
    ? [...sweep.points]
        .filter((p) => p.epsilon !== null)
        .sort((a, b) => (a.epsilon as number) - (b.epsilon as number))
        .map((p) => ({ ...p, epsilonLabel: String(p.epsilon) }))
    : []
  const noDpPoint = sweep?.points.find((p) => p.epsilon === null)
  if (noDpPoint && chartData.length) {
    const ceilingX = (chartData[chartData.length - 1].epsilon as number) * 4
    chartData.push({ ...noDpPoint, epsilon: ceilingX, epsilonLabel: "no DP" })
  }

  return (
    <div>
      <SectionTitle
        eyebrow="Honest accounting"
        title="Architecture & Privacy"
        sub="What's actually wired into training vs. what's a documented simplification or reference implementation — no rounding up."
      />

      <Reveal><Card className="mb-6">
        <h3 className="font-bold text-lg mb-4">Privacy / security layers</h3>
        <div className="space-y-3">
          {LAYERS.map((l) => (
            <div key={l.name} className="flex items-center justify-between border-b border-white/5 pb-3 last:border-0 last:pb-0">
              <div>
                <div className="font-semibold text-sm">{l.name}</div>
                <div className="text-xs text-ink-muted">{l.guarantee}</div>
              </div>
              <Badge status={l.status}>{l.label}</Badge>
            </div>
          ))}
        </div>
      </Card></Reveal>

      <Reveal delay={0.08}><Card className="mb-6">
        <h3 className="font-bold text-lg mb-2">Differential privacy — the real privacy/utility tradeoff curve</h3>
        <p className="text-sm text-ink-secondary mb-4">
          The same FedProx-federated model, same rounds, same held-out test set, only ε changing — this
          is the standard evidence that the DP mechanism has a real, measurable effect (not a config
          flag that does nothing): accuracy visibly degrades as privacy gets stronger (lower ε), with a
          clear collapse below ε≈50. This project's default, <span className="font-mono">ε=100</span>,
          is marked below — a deliberate, measured choice, not an arbitrary one. Produced by{" "}
          <span className="font-mono text-xs">experiments/run_epsilon_sweep.py</span>.
        </p>

        {sweepError && (
          <div className="text-status-critical text-sm">
            Couldn't load the sweep: {sweepError}. Run{" "}
            <span className="font-mono text-xs">python experiments/run_epsilon_sweep.py</span> from the repo root.
          </div>
        )}

        {!sweep && !sweepError && (
          <div className="text-ink-muted text-sm">Loading…</div>
        )}

        {sweep && chartData.length > 0 && (
          <div className="h-72">
            <ResponsiveContainer width="100%" height="100%">
              <LineChart data={chartData} margin={{ left: 4, right: 12, top: 8 }}>
                <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" />
                <XAxis
                  dataKey="epsilon"
                  scale="log"
                  domain={["dataMin", "dataMax"]}
                  type="number"
                  tickFormatter={(v: number) => {
                    const p = chartData.find((d) => d.epsilon === v)
                    return p?.epsilonLabel ?? String(v)
                  }}
                  ticks={chartData.map((d) => d.epsilon as number)}
                  tick={{ fontSize: 11, fill: "#8b8a83" }}
                  axisLine={false}
                  tickLine={false}
                  label={{ value: "ε (privacy budget, log scale — smaller = more private)", position: "insideBottom", offset: -4, fontSize: 11, fill: "#8b8a83" }}
                />
                <YAxis
                  domain={[0, 1]}
                  tickFormatter={(v: number) => `${(v * 100).toFixed(0)}%`}
                  tick={{ fontSize: 11, fill: "#8b8a83" }}
                  axisLine={false}
                  tickLine={false}
                />
                <Tooltip
                  contentStyle={{ background: "#1a1a19", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }}
                  labelFormatter={((v: number) => {
                    const p = chartData.find((d) => d.epsilon === v)
                    return `ε = ${p?.epsilonLabel ?? v}`
                  }) as never}
                  formatter={((v: number, name: string) => [`${(v * 100).toFixed(1)}%`, name]) as never}
                />
                <Legend wrapperStyle={{ fontSize: 12 }} />
                <ReferenceLine
                  x={DEFAULT_EPSILON}
                  stroke="#3987e5"
                  strokeDasharray="4 4"
                  label={{ value: "default (ε=100)", position: "top", fontSize: 11, fill: "#3987e5" }}
                />
                <Line type="monotone" dataKey="accuracy" name="Accuracy" stroke="#3987e5" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="f1" name="F1" stroke="#9085e9" strokeWidth={2} dot={{ r: 3 }} />
                <Line type="monotone" dataKey="recall" name="Recall" stroke="#199e70" strokeWidth={2} dot={{ r: 3 }} strokeDasharray="3 3" />
              </LineChart>
            </ResponsiveContainer>
          </div>
        )}
      </Card></Reveal>

      <Stagger className="grid md:grid-cols-2 gap-6">
        <StaggerItem><Card>
          <h3 className="font-bold text-lg mb-2">The graph branch</h3>
          <p className="text-sm text-ink-secondary">
            The real heterogeneous graph encoder (HGTConv, 3 layers/8 heads) is fully implemented but
            needs real entity-linked edges (shared user/merchant/card/device) that the preprocessed
            sequence data doesn't carry. What runs today is a simplified per-sample encoder over each
            account's own transaction profile — real signal, not zeros, but not a relational graph.
          </p>
        </Card></StaggerItem>
        <StaggerItem><Card>
          <h3 className="font-bold text-lg mb-2">Where the data comes from</h3>
          <p className="text-sm text-ink-secondary">
            The original PaySim dataset isn't available on this machine. Training here runs on a
            rule-based synthetic stand-in with the same schema (see <span className="font-mono text-xs">experiments/generate_synthetic_paysim.py</span>) —
            useful as a pipeline-correctness check, not a real-world performance claim.
          </p>
        </Card></StaggerItem>
      </Stagger>
    </div>
  )
}
