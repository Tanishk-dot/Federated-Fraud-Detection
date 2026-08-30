import { motion } from "framer-motion"
import { type MouseEvent, type ReactNode, useRef, useState } from "react"
import { Link } from "react-router-dom"
import { EASE, Reveal, useMagnetic } from "./motion"

const MotionLink = motion.create(Link)

/**
 * A soft radial glow that tracks the cursor - invisible at rest, appears
 * only on hover. The Motion Primitives "Spotlight" pattern, folded
 * directly into Card so every card in the app gets it for free.
 */
export function Card({
  children, className = "",
}: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const [pos, setPos] = useState({ x: 50, y: 50 })
  const [hovered, setHovered] = useState(false)

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    setPos({ x: ((e.clientX - rect.left) / rect.width) * 100, y: ((e.clientY - rect.top) / rect.height) * 100 })
  }

  return (
    <div
      ref={ref}
      onMouseMove={onMove}
      onMouseEnter={() => setHovered(true)}
      onMouseLeave={() => setHovered(false)}
      className={`relative overflow-hidden rounded-xl border border-white/8 bg-surface-2/70 p-6 transition-colors duration-300 hover:border-white/16 ${className}`}
    >
      <div
        className="pointer-events-none absolute inset-0 transition-opacity duration-500"
        style={{
          opacity: hovered ? 1 : 0,
          background: `radial-gradient(360px circle at ${pos.x}% ${pos.y}%, rgba(57,135,229,0.08), transparent 70%)`,
        }}
      />
      <div className="relative">{children}</div>
    </div>
  )
}

export function SectionTitle({
  eyebrow, title, sub, index,
}: { eyebrow?: string; title: string; sub?: string; index?: string }) {
  return (
    <Reveal className="mb-9">
      {eyebrow && (
        <div className="flex items-center gap-2.5 mb-3">
          {index && <span className="text-xs font-mono text-ink-muted">{index}</span>}
          <span className="h-px w-6 bg-series-blue/60" />
          <span className="text-xs font-semibold uppercase tracking-[0.14em] text-series-blue">{eyebrow}</span>
        </div>
      )}
      <h1 className="text-[2.1rem] leading-[1.08] md:text-[2.6rem] font-extrabold tracking-tight text-ink-primary">
        {title}
      </h1>
      {sub && <p className="text-ink-secondary mt-3 max-w-2xl text-[15px] leading-relaxed">{sub}</p>}
    </Reveal>
  )
}

const statusColor: Record<string, string> = {
  good: "text-status-good border-status-good/25 bg-status-good/8",
  warning: "text-status-warning border-status-warning/25 bg-status-warning/8",
  serious: "text-status-serious border-status-serious/25 bg-status-serious/8",
  critical: "text-status-critical border-status-critical/25 bg-status-critical/8",
  neutral: "text-ink-secondary border-white/12 bg-white/4",
}

export function Badge({ status = "neutral", children }: { status?: keyof typeof statusColor; children: ReactNode }) {
  return (
    <span className={`inline-flex items-center gap-1.5 rounded-full border px-2.5 py-1 text-xs font-medium ${statusColor[status]}`}>
      {children}
    </span>
  )
}

// Tailwind's scanner needs full literal class strings in source - a dynamic
// `text-${accent}` template won't be picked up, so map to complete strings.
const ACCENTS = {
  blue: "text-series-blue",
  violet: "text-series-violet",
  aqua: "text-series-aqua",
  good: "text-status-good",
  warning: "text-status-warning",
  critical: "text-status-critical",
} as const

export function StatTile({
  label, value, sub, accent = "blue",
}: { label: string; value: string; sub?: string; accent?: keyof typeof ACCENTS }) {
  return (
    <Card className="p-5">
      <div className="text-xs font-medium uppercase tracking-wide text-ink-muted mb-2">{label}</div>
      <div className={`text-3xl font-extrabold tabular-nums ${ACCENTS[accent]}`}>{value}</div>
      {sub && <div className="text-sm text-ink-secondary mt-1">{sub}</div>}
    </Card>
  )
}

export function Button({
  children, onClick, variant = "primary", disabled, type = "button", className = "", to,
}: {
  children: ReactNode; onClick?: () => void; variant?: "primary" | "ghost" | "danger"
  disabled?: boolean; type?: "button" | "submit"; className?: string; to?: string
}) {
  // Magnetic nudge-toward-cursor, reserved for primary CTAs so it reads as
  // a deliberate detail rather than a page-wide tic.
  const magnetic = useMagnetic(variant === "primary" && !disabled ? 0.25 : 0)
  const base = "rounded-lg px-4 py-2.5 font-semibold text-sm disabled:opacity-40 disabled:cursor-not-allowed inline-flex items-center justify-center"
  const variants = {
    primary: "bg-series-blue text-white shadow-[0_1px_0_rgba(255,255,255,0.15)_inset,0_2px_12px_rgba(57,135,229,0.35)]",
    ghost: "bg-white/5 border border-white/10 text-ink-primary",
    danger: "bg-status-critical/12 border border-status-critical/35 text-status-critical",
  }
  const shared = {
    className: `${base} ${variants[variant]} ${className}`,
    whileHover: disabled ? undefined : { y: -1, filter: "brightness(1.08)" },
    whileTap: disabled ? undefined : { scale: 0.97, y: 0 },
    transition: { duration: 0.15, ease: EASE },
    style: variant === "primary" ? magnetic.style : undefined,
    onMouseMove: variant === "primary" ? magnetic.onMouseMove : undefined,
    onMouseLeave: variant === "primary" ? magnetic.onMouseLeave : undefined,
  }

  if (to) {
    return (
      <MotionLink to={to} ref={magnetic.ref as never} {...shared}>
        {children}
      </MotionLink>
    )
  }
  return (
    <motion.button
      type={type}
      onClick={onClick}
      disabled={disabled}
      ref={variant === "primary" ? (magnetic.ref as never) : undefined}
      {...shared}
    >
      {children}
    </motion.button>
  )
}

export function ProgressBar({ value }: { value: number }) {
  return (
    <div className="h-1.5 w-full rounded-full bg-white/8 overflow-hidden">
      <motion.div
        className="h-full rounded-full bg-series-blue"
        animate={{ width: `${Math.min(100, Math.max(0, value * 100))}%` }}
        transition={{ duration: 0.4, ease: EASE }}
      />
    </div>
  )
}

/**
 * Editorial metric row: a horizontal strip of numbers divided by hairlines
 * instead of a grid of boxed stat-card - the generic-SaaS "four cards in a
 * row" pattern, avoided.
 */
export function MetricRow({
  items,
}: { items: { label: string; value: ReactNode; sub?: string }[] }) {
  return (
    <div className="flex flex-wrap divide-x divide-white/8 border-y border-white/8 -mx-1">
      {items.map((it, i) => (
        <Reveal key={it.label} delay={i * 0.06} className="flex-1 min-w-[9rem] px-6 py-5">
          <div className="text-[11px] font-medium uppercase tracking-wide text-ink-muted mb-1.5">{it.label}</div>
          <div className="text-3xl font-extrabold tabular-nums text-ink-primary">{it.value}</div>
          {it.sub && <div className="text-xs text-ink-muted mt-1">{it.sub}</div>}
        </Reveal>
      ))}
    </div>
  )
}
