import { useEffect, useState } from "react"
import { Link } from "react-router-dom"
import { api, type ModelInfo } from "../api"
import { Badge, Button, Card, MetricRow } from "../components/ui"
import { AmbientField, CountUp, Reveal, SplitText, Stagger, StaggerItem, TiltCard } from "../components/motion"

const PIPELINE: { stage: string; detail: string; note?: string }[] = [
  { stage: "Temporal branch", detail: "4-layer Transformer over the last 10 transactions" },
  { stage: "Graph branch", detail: "Per-sample encoder over each account's own profile", note: "simplified" },
  { stage: "Cross-modal fusion", detail: "Bidirectional attention between both branches" },
  { stage: "Dual heads", detail: "Supervised (BCE) + contrastive (novel-fraud detection)" },
]

const PRIVACY_STACK: { label: string; status: "good" | "warning" | "neutral"; note: string }[] = [
  { label: "Data locality", status: "good", note: "real" },
  { label: "Client-level DP-FedAvg", status: "good", note: "real" },
  { label: "Homomorphic encryption", status: "warning", note: "implemented, not wired" },
  { label: "TLS / mutual auth", status: "neutral", note: "n/a (single-process sim)" },
]

function SectionLabel({ index, title }: { index: string; title: string }) {
  return (
    <Reveal className="mb-3 flex items-center gap-2.5">
      <span className="text-xs font-mono text-ink-muted">{index}</span>
      <span className="h-px w-6 bg-white/20" />
      <h2 className="text-xs font-semibold uppercase tracking-[0.14em] text-ink-secondary">{title}</h2>
    </Reveal>
  )
}

export default function Overview() {
  const [info, setInfo] = useState<ModelInfo | null>(null)
  const [error, setError] = useState<string | null>(null)

  useEffect(() => {
    api.modelInfo().then(setInfo).catch((e) => setError(String(e)))
  }, [])

  const m = info?.final_metrics as Record<string, number> | undefined

  return (
    <div>
      {error && (
        <Card className="mb-6 border-status-critical/30">
          <div className="text-status-critical text-sm">Couldn't reach the backend API: {error}</div>
        </Card>
      )}

      {info && !info.loaded && (
        <Card className="mb-6">
          <div className="text-ink-secondary text-sm">{info.message}</div>
        </Card>
      )}

      <div className={`relative grid gap-10 items-end pt-2 mb-10 ${m ? "lg:grid-cols-[1.3fr_1fr]" : ""}`}>
        <AmbientField className="inset-0 -z-10" />

        <Reveal>
          <div className="flex items-center gap-2.5 mb-4">
            <span className="h-px w-6 bg-series-blue/60" />
            <span className="text-xs font-semibold uppercase tracking-[0.14em] text-series-blue">
              Federated · Privacy-preserving
            </span>
          </div>
          <h1 className="text-[2.4rem] leading-[1.05] md:text-[3.2rem] font-extrabold tracking-tight text-ink-primary">
            <SplitText text="Fraud detection that" />{" "}
            <SplitText text="never" delay={0.2} wordClassName="font-display text-series-blue" />{" "}
            <SplitText text="sees your raw data." delay={0.26} />
          </h1>
          <p className="text-ink-secondary mt-5 max-w-xl text-[15px] leading-relaxed">
            A Temporal Graph Transformer with contrastive learning, trained across simulated banks
            with FedProx and client-level differential privacy — every number on this page comes
            from an actual trained checkpoint, not a projection.
          </p>
          <div className="flex items-center gap-3 mt-7">
            <Button to="/test-data" variant="primary">Test your data →</Button>
            <Button to="/architecture" variant="ghost">How it's private</Button>
          </div>
        </Reveal>

        {m && (
          <Reveal delay={0.15} className="lg:text-right">
            <div className="text-xs font-medium uppercase tracking-wide text-ink-muted mb-1">
              Recall on held-out fraud
            </div>
            <div className="text-[4rem] leading-none md:text-[4.5rem] font-extrabold tabular-nums text-series-blue">
              <CountUp value={m.recall * 100} decimals={1} suffix="%" />
            </div>
            <div className="text-sm text-ink-secondary mt-2">real, measured — not illustrative</div>
          </Reveal>
        )}
      </div>

      {m && (
        <div className="mb-14">
          <MetricRow items={[
            { label: "Accuracy", value: `${(m.accuracy * 100).toFixed(1)}%` },
            { label: "F1 score", value: `${(m.f1 * 100).toFixed(1)}%` },
            { label: "ROC-AUC", value: `${(m.roc_auc * 100).toFixed(1)}%` },
            { label: "Recall", value: `${(m.recall * 100).toFixed(1)}%` },
          ]} />
        </div>
      )}

      {info?.loaded && (
        <Reveal className="mb-14">
          <p className="text-xs text-ink-muted max-w-3xl">
            From <span className="font-mono">{info.checkpoint}</span> — {info.num_rounds} real training
            rounds, on rule-based synthetic data (the original PaySim dataset isn't available on this
            machine). See{" "}
            <Link to="/architecture" className="text-series-blue underline underline-offset-2">Architecture</Link>{" "}
            for what that means for these numbers, or{" "}
            <Link to="/comparison" className="text-series-blue underline underline-offset-2">Model Comparison</Link>{" "}
            for how this stacks up against real baselines.
          </p>
        </Reveal>
      )}

      <SectionLabel index="01" title="Architecture" />
      <Stagger className="grid sm:grid-cols-2 lg:grid-cols-4 gap-px bg-white/8 rounded-xl overflow-hidden mb-14 border border-white/8">
        {PIPELINE.map((p, i) => (
          <StaggerItem key={p.stage}>
            <TiltCard className="h-full bg-surface-2 p-5">
              <div className="text-[11px] font-mono text-ink-muted mb-2.5">{String(i + 1).padStart(2, "0")}</div>
              <div className="font-semibold text-sm text-ink-primary mb-1.5 flex flex-wrap items-center gap-2">
                {p.stage}
                {p.note && <Badge status="warning">{p.note}</Badge>}
              </div>
              <div className="text-xs text-ink-secondary leading-relaxed">{p.detail}</div>
            </TiltCard>
          </StaggerItem>
        ))}
      </Stagger>

      <SectionLabel index="02" title="Privacy stack" />
      <Stagger className="divide-y divide-white/8 border-y border-white/8 mb-4">
        {PRIVACY_STACK.map((p) => (
          <StaggerItem key={p.label} className="flex items-center justify-between py-3.5">
            <span className="text-sm text-ink-secondary">{p.label}</span>
            <Badge status={p.status}>{p.note}</Badge>
          </StaggerItem>
        ))}
      </Stagger>
    </div>
  )
}
