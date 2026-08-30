import { useEffect, useRef, useState } from "react"
import {
  CartesianGrid, Line, LineChart, ResponsiveContainer, Tooltip, XAxis, YAxis,
} from "recharts"
import { api, type DatasetInfo, type TrainJobStatus } from "../api"
import { Badge, Button, Card, ProgressBar, SectionTitle } from "../components/ui"
import { Reveal } from "../components/motion"

export default function Train() {
  const [datasets, setDatasets] = useState<DatasetInfo[]>([])
  const [dataset, setDataset] = useState("")
  const [rounds, setRounds] = useState(8)
  const [clientsPerRound, setClientsPerRound] = useState(3)
  const [hiddenDim, setHiddenDim] = useState(32)
  const [dpEnabled, setDpEnabled] = useState(false)
  const [epsilon, setEpsilon] = useState(100)

  const [jobId, setJobId] = useState<string | null>(null)
  const [job, setJob] = useState<TrainJobStatus | null>(null)
  const [error, setError] = useState<string | null>(null)
  const pollRef = useRef<number | null>(null)

  useEffect(() => {
    api.datasets().then((r) => {
      setDatasets(r.datasets)
      if (r.datasets[0]) setDataset(r.datasets[0].name)
    })
  }, [])

  useEffect(() => {
    if (!jobId) return
    const poll = async () => {
      const status = await api.trainStatus(jobId)
      setJob(status)
      if (status.status === "running" || status.status === "starting") {
        pollRef.current = window.setTimeout(poll, 800)
      }
    }
    poll()
    return () => { if (pollRef.current) clearTimeout(pollRef.current) }
  }, [jobId])

  const start = async () => {
    setError(null); setJob(null)
    try {
      const { job_id } = await api.trainStart({
        dataset, rounds, clients_per_round: clientsPerRound,
        hidden_dim: hiddenDim, max_samples: 600, dp_enabled: dpEnabled, epsilon,
      })
      setJobId(job_id)
    } catch (e) {
      setError(String(e))
    }
  }

  const running = job?.status === "running" || job?.status === "starting"

  return (
    <div>
      <SectionTitle
        eyebrow="Real federated training"
        title="Train"
        sub="Runs an actual (small, fast) FedProx round loop on the selected dataset, using the real model and the real client-level DP-FedAvg mechanism — not a simulation of a simulation."
      />

      <div className="grid md:grid-cols-3 gap-6">
        <Reveal className="md:col-span-1"><Card className="h-fit">
          <h3 className="font-bold mb-4">Configuration</h3>

          <label className="flex flex-col gap-1 text-sm mb-4">
            <span className="text-ink-muted">Dataset</span>
            {datasets.length === 0 ? (
              <span className="text-status-warning text-xs">
                None found under data/ — run experiments/generate_synthetic_paysim.py first.
              </span>
            ) : (
              <select value={dataset} onChange={(e) => setDataset(e.target.value)}
                className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 focus:outline-none focus:border-series-blue">
                {datasets.map((d) => <option key={d.name} value={d.name}>{d.name} ({d.num_clients} clients)</option>)}
              </select>
            )}
          </label>

          <label className="flex flex-col gap-1 text-sm mb-4">
            <span className="text-ink-muted">FL rounds: {rounds}</span>
            <input type="range" min={3} max={20} value={rounds} onChange={(e) => setRounds(+e.target.value)} className="accent-[#3987e5]" />
          </label>

          <label className="flex flex-col gap-1 text-sm mb-4">
            <span className="text-ink-muted">Clients sampled per round: {clientsPerRound}</span>
            <input type="range" min={1} max={5} value={clientsPerRound} onChange={(e) => setClientsPerRound(+e.target.value)} className="accent-[#3987e5]" />
          </label>

          <label className="flex flex-col gap-1 text-sm mb-4">
            <span className="text-ink-muted">Hidden dimension</span>
            <select value={hiddenDim} onChange={(e) => setHiddenDim(+e.target.value)}
              className="rounded-lg bg-white/5 border border-white/10 px-3 py-2 focus:outline-none focus:border-series-blue">
              {[16, 32, 64, 128].map((d) => <option key={d} value={d}>{d}</option>)}
            </select>
          </label>

          <label className="flex items-center gap-2 text-sm mb-3">
            <input type="checkbox" checked={dpEnabled} onChange={(e) => setDpEnabled(e.target.checked)} className="accent-[#3987e5]" />
            <span>Enable differential privacy</span>
          </label>

          {dpEnabled && (
            <label className="flex flex-col gap-1 text-sm mb-4">
              <span className="text-ink-muted">Privacy budget ε: {epsilon}</span>
              <input type="range" min={20} max={500} step={10} value={epsilon} onChange={(e) => setEpsilon(+e.target.value)} className="accent-[#3987e5]" />
              <span className="text-xs text-ink-muted">
                ε=1.0 (textbook) still destroys this model — verified, see Architecture page. ~50 is where it starts working; ~100 gives working margin.
              </span>
            </label>
          )}

          <Button onClick={start} disabled={running || datasets.length === 0} className="w-full mt-2">
            {running ? "Training..." : "🚀 Start Training"}
          </Button>
          {error && <p className="text-status-critical text-sm mt-3">{error}</p>}
        </Card></Reveal>

        <Reveal delay={0.1} className="md:col-span-2"><Card>
          {!job && <p className="text-ink-muted text-sm">Configure and start a run — progress appears here live.</p>}

          {job && (
            <div>
              <div className="flex items-center justify-between mb-4">
                <div className="flex items-center gap-3">
                  {job.status === "running" || job.status === "starting" ? (
                    <Badge status="warning">⏳ training in progress</Badge>
                  ) : job.status === "complete" ? (
                    <Badge status="good">✅ complete</Badge>
                  ) : (
                    <Badge status="critical">✗ error</Badge>
                  )}
                  <span className="text-sm text-ink-muted">Round {job.round}/{job.total_rounds}</span>
                </div>
              </div>

              <ProgressBar value={job.total_rounds ? job.round / job.total_rounds : 0} />

              {job.error && <p className="text-status-critical text-sm mt-4">{job.error}</p>}

              {job.history.length > 0 && (
                <>
                  <div className="h-56 mt-6">
                    <ResponsiveContainer width="100%" height="100%">
                      <LineChart data={job.history}>
                        <CartesianGrid strokeDasharray="3 3" stroke="rgba(255,255,255,0.06)" vertical={false} />
                        <XAxis dataKey="round" tick={{ fontSize: 11, fill: "#8b8a83" }} axisLine={{ stroke: "rgba(255,255,255,0.1)" }} tickLine={false} />
                        <YAxis domain={[0, 1]} tick={{ fontSize: 11, fill: "#8b8a83" }} axisLine={false} tickLine={false} />
                        <Tooltip contentStyle={{ background: "#1a1a19", border: "1px solid rgba(255,255,255,0.1)", borderRadius: 8 }} />
                        <Line type="monotone" dataKey="accuracy" stroke="#3987e5" strokeWidth={2} dot={false} name="Accuracy" />
                        <Line type="monotone" dataKey="f1" stroke="#9085e9" strokeWidth={2} dot={false} name="F1" />
                        <Line type="monotone" dataKey="roc_auc" stroke="#199e70" strokeWidth={2} dot={false} name="ROC-AUC" />
                      </LineChart>
                    </ResponsiveContainer>
                  </div>

                  <div className="mt-4 max-h-56 overflow-y-auto rounded-xl border border-white/8">
                    <table className="w-full text-sm">
                      <thead className="sticky top-0 bg-surface-2 text-ink-muted text-xs uppercase">
                        <tr>
                          <th className="text-left px-3 py-2">Round</th>
                          <th className="text-left px-3 py-2">Clients</th>
                          <th className="text-left px-3 py-2">Accuracy</th>
                          <th className="text-left px-3 py-2">F1</th>
                          <th className="text-left px-3 py-2">ROC-AUC</th>
                        </tr>
                      </thead>
                      <tbody>
                        {[...job.history].reverse().map((r) => (
                          <tr key={r.round} className="border-t border-white/5">
                            <td className="px-3 py-2 tabular-nums">{r.round}</td>
                            <td className="px-3 py-2 text-ink-muted text-xs">{r.selected_clients.join(", ")}</td>
                            <td className="px-3 py-2 tabular-nums">{(r.accuracy * 100).toFixed(1)}%</td>
                            <td className="px-3 py-2 tabular-nums">{(r.f1 * 100).toFixed(1)}%</td>
                            <td className="px-3 py-2 tabular-nums">{(r.roc_auc * 100).toFixed(1)}%</td>
                          </tr>
                        ))}
                      </tbody>
                    </table>
                  </div>
                </>
              )}
            </div>
          )}
        </Card></Reveal>
      </div>
    </div>
  )
}
