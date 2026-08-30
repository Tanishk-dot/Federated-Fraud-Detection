import { AnimatePresence, motion } from "framer-motion"
import { ReactLenis } from "lenis/react"
import { useEffect, useState } from "react"
import { NavLink, Outlet, useLocation } from "react-router-dom"
import { api } from "../api"
import { EASE } from "./motion"

const NAV = [
  { to: "/", label: "Overview", index: "01" },
  { to: "/test-data", label: "Test Your Data", index: "02" },
  { to: "/train", label: "Train", index: "03" },
  { to: "/simulation", label: "Live Simulation", index: "04" },
  { to: "/comparison", label: "Model Comparison", index: "05" },
  { to: "/architecture", label: "Architecture & Privacy", index: "06" },
]

function Wordmark() {
  return (
    <div className="flex items-center gap-2.5">
      <span className="relative flex h-6 w-6 items-center justify-center rounded-md bg-series-blue/15">
        <span className="h-2 w-2 rounded-[2px] bg-series-blue" />
      </span>
      <div>
        <div className="text-[13px] font-bold tracking-tight text-ink-primary leading-none">Fraud Detection</div>
        <div className="text-[10px] text-ink-muted uppercase tracking-widest mt-1">Federated · Private</div>
      </div>
    </div>
  )
}

function StatusDot() {
  const [ok, setOk] = useState<boolean | null>(null)
  useEffect(() => {
    let alive = true
    const check = () => api.health().then(() => alive && setOk(true)).catch(() => alive && setOk(false))
    check()
    const id = window.setInterval(check, 20000)
    return () => { alive = false; window.clearInterval(id) }
  }, [])
  return (
    <div className="flex items-center gap-2 text-[11px] text-ink-muted">
      <span className="relative flex h-1.5 w-1.5">
        {ok && <span className="absolute inline-flex h-full w-full animate-ping rounded-full bg-status-good/60" />}
        <span className={`relative inline-flex h-1.5 w-1.5 rounded-full ${ok === null ? "bg-ink-muted" : ok ? "bg-status-good" : "bg-status-critical"}`} />
      </span>
      {ok === null ? "checking API…" : ok ? "API connected" : "API unreachable"}
    </div>
  )
}

function NavItems({ onNavigate }: { onNavigate?: () => void }) {
  return (
    <nav className="flex flex-col gap-1">
      {NAV.map((item) => (
        <NavLink
          key={item.to}
          to={item.to}
          end={item.to === "/"}
          onClick={onNavigate}
          className="relative flex items-center gap-3 rounded-lg px-3 py-2.5 text-sm font-medium text-ink-secondary transition-colors hover:text-white"
        >
          {({ isActive }) => (
            <>
              {isActive && (
                <motion.span
                  layoutId="nav-active-pill"
                  className="absolute inset-0 rounded-lg bg-series-blue/12 border border-series-blue/25"
                  transition={{ duration: 0.35, ease: EASE }}
                />
              )}
              <span className={`relative z-10 font-mono text-[11px] ${isActive ? "text-series-blue" : "text-ink-muted"}`}>
                {item.index}
              </span>
              <span className={`relative z-10 ${isActive ? "text-white" : ""}`}>{item.label}</span>
            </>
          )}
        </NavLink>
      ))}
    </nav>
  )
}

/** A slim top-of-viewport bar that sweeps in on every route change - the
 * Linear/Vercel-style "something just happened" signature. */
function RouteProgress() {
  const { pathname } = useLocation()
  return (
    <AnimatePresence>
      <motion.div
        key={pathname}
        className="fixed top-0 left-0 right-0 z-50 h-[2px] origin-left bg-series-blue"
        initial={{ scaleX: 0, opacity: 1 }}
        animate={{ scaleX: 1, transition: { duration: 0.5, ease: EASE } }}
        exit={{ opacity: 0, transition: { duration: 0.25 } }}
      />
    </AnimatePresence>
  )
}

export default function Layout() {
  const [mobileOpen, setMobileOpen] = useState(false)
  const location = useLocation()

  useEffect(() => { setMobileOpen(false) }, [location.pathname])

  return (
    <ReactLenis root options={{ duration: 1.1, smoothWheel: true, syncTouch: false }}>
      <RouteProgress />
      <div className="min-h-screen md:flex">
        {/* Mobile top bar */}
        <header className="md:hidden sticky top-0 z-30 flex items-center justify-between border-b border-white/8 bg-surface-1/95 backdrop-blur px-4 py-3">
          <Wordmark />
          <button
            onClick={() => setMobileOpen((v) => !v)}
            aria-label="Toggle navigation"
            className="flex h-9 w-9 items-center justify-center rounded-lg border border-white/10 text-ink-secondary"
          >
            <span className="text-lg leading-none">{mobileOpen ? "✕" : "☰"}</span>
          </button>
        </header>

        <AnimatePresence>
          {mobileOpen && (
            <motion.div
              initial={{ opacity: 0 }}
              animate={{ opacity: 1 }}
              exit={{ opacity: 0 }}
              transition={{ duration: 0.2 }}
              className="md:hidden fixed inset-0 z-20 bg-black/60"
              onClick={() => setMobileOpen(false)}
            />
          )}
        </AnimatePresence>
        <AnimatePresence>
          {mobileOpen && (
            <motion.aside
              initial={{ x: "-100%" }}
              animate={{ x: 0 }}
              exit={{ x: "-100%" }}
              transition={{ duration: 0.3, ease: EASE }}
              className="md:hidden fixed inset-y-0 left-0 z-30 w-72 bg-surface-1 border-r border-white/8 px-4 py-6 flex flex-col gap-6"
            >
              <div className="px-2"><Wordmark /></div>
              <NavItems onNavigate={() => setMobileOpen(false)} />
              <div className="mt-auto px-2 flex flex-col gap-3">
                <StatusDot />
                <p className="text-[11px] text-ink-muted leading-relaxed">
                  Real trained model · real training runs · no fabricated metrics.
                </p>
              </div>
            </motion.aside>
          )}
        </AnimatePresence>

        {/* Desktop sidebar */}
        <aside className="hidden md:flex w-64 shrink-0 border-r border-white/8 bg-surface-1 px-4 py-6 flex-col gap-6 sticky top-0 h-screen">
          <div className="px-2"><Wordmark /></div>
          <NavItems />
          <div className="mt-auto px-2 flex flex-col gap-3">
            <StatusDot />
            <p className="text-[11px] text-ink-muted leading-relaxed">
              Real trained model · real training runs · no fabricated metrics.
            </p>
          </div>
        </aside>

        <main className="flex-1 min-w-0 px-5 py-8 md:px-10">
          <div className="max-w-6xl mx-auto">
            <AnimatePresence mode="wait">
              <motion.div
                key={location.pathname}
                initial={{ opacity: 0, y: 8 }}
                animate={{ opacity: 1, y: 0 }}
                exit={{ opacity: 0 }}
                transition={{ duration: 0.35, ease: EASE }}
              >
                <Outlet />
              </motion.div>
            </AnimatePresence>
          </div>
        </main>
      </div>
    </ReactLenis>
  )
}
