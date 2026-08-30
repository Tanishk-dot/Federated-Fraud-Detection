import { motion, useMotionValue, useSpring, type Variants } from "framer-motion"
import { type MouseEvent, type ReactNode, useEffect, useRef, useState } from "react"

export const EASE = [0.22, 1, 0.36, 1] as const

const revealVariants: Variants = {
  hidden: { opacity: 0, y: 16 },
  show: { opacity: 1, y: 0 },
}

/**
 * Fades + slides a section up once, the moment it enters the viewport.
 * The one motion primitive used everywhere a block of content appears -
 * one signature instead of a different animation per page.
 */
export function Reveal({
  children, delay = 0, className = "", as = "div",
}: { children: ReactNode; delay?: number; className?: string; as?: "div" | "span" }) {
  const Component = motion[as]
  return (
    <Component
      className={className}
      variants={revealVariants}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      transition={{ duration: 0.6, ease: EASE, delay }}
    >
      {children}
    </Component>
  )
}

/**
 * A container that staggers its direct Reveal children in sequence -
 * for lists/grids that should read as one composed reveal, not N
 * independent ones firing at once.
 */
export function Stagger({
  children, className = "", gap = 0.08,
}: { children: ReactNode; className?: string; gap?: number }) {
  return (
    <motion.div
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      transition={{ staggerChildren: gap }}
    >
      {children}
    </motion.div>
  )
}

export function StaggerItem({ children, className = "" }: { children: ReactNode; className?: string }) {
  return (
    <motion.div className={className} variants={revealVariants} transition={{ duration: 0.5, ease: EASE }}>
      {children}
    </motion.div>
  )
}

/** Tween a number up from 0 the moment it scrolls into view - used
 * sparingly, on the one or two headline metrics that deserve it. */
export function CountUp({
  value, decimals = 1, suffix = "", duration = 1.1,
}: { value: number; decimals?: number; suffix?: string; duration?: number }) {
  const ref = useRef<HTMLSpanElement>(null)
  const [display, setDisplay] = useState(0)
  const [started, setStarted] = useState(false)

  useEffect(() => {
    const el = ref.current
    if (!el || started) return
    const io = new IntersectionObserver(
      ([entry]) => {
        if (!entry.isIntersecting) return
        setStarted(true)
        const t0 = performance.now()
        const tick = (now: number) => {
          const p = Math.min(1, (now - t0) / (duration * 1000))
          const eased = 1 - Math.pow(1 - p, 3)
          setDisplay(value * eased)
          if (p < 1) requestAnimationFrame(tick)
        }
        requestAnimationFrame(tick)
        io.disconnect()
      },
      { threshold: 0.4 },
    )
    io.observe(el)
    return () => io.disconnect()
    // eslint-disable-next-line react-hooks/exhaustive-deps
  }, [value])

  return (
    <span ref={ref} className="tabular-nums">
      {display.toFixed(decimals)}{suffix}
    </span>
  )
}

const wordVariants: Variants = {
  hidden: { opacity: 0, y: 14, filter: "blur(6px)" },
  show: { opacity: 1, y: 0, filter: "blur(0px)" },
}

/** Word-by-word blur+slide reveal for a headline - the Motion Primitives
 * "text effect" pattern. Used once, on the one headline that should carry
 * it - not a global text treatment. */
export function SplitText({
  text, className = "", delay = 0, wordClassName = "",
}: { text: string; className?: string; delay?: number; wordClassName?: string }) {
  const words = text.split(" ")
  return (
    <motion.span
      className={className}
      initial="hidden"
      whileInView="show"
      viewport={{ once: true, margin: "-60px" }}
      transition={{ staggerChildren: 0.045, delayChildren: delay }}
    >
      {words.map((w, i) => (
        <motion.span
          key={i}
          className={`inline-block ${wordClassName}`}
          variants={wordVariants}
          transition={{ duration: 0.55, ease: EASE }}
        >
          {w}
          {i < words.length - 1 ? " " : ""}
        </motion.span>
      ))}
    </motion.span>
  )
}

/**
 * Cursor-attraction for a button/link - the element nudges toward the
 * pointer within its own bounds, springs back on leave. Applied narrowly
 * (primary CTAs only) so it reads as a deliberate tactile detail, not a
 * page-wide gimmick.
 */
export function useMagnetic(strength = 0.3) {
  const ref = useRef<HTMLElement>(null)
  const x = useMotionValue(0)
  const y = useMotionValue(0)
  const springX = useSpring(x, { stiffness: 150, damping: 14, mass: 0.2 })
  const springY = useSpring(y, { stiffness: 150, damping: 14, mass: 0.2 })

  const onMouseMove = (e: MouseEvent) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect || strength === 0) return
    x.set((e.clientX - rect.left - rect.width / 2) * strength)
    y.set((e.clientY - rect.top - rect.height / 2) * strength)
  }
  const onMouseLeave = () => { x.set(0); y.set(0) }

  return { ref, style: { x: springX, y: springY }, onMouseMove, onMouseLeave }
}

/** Subtle cursor-following 3D tilt, for the one grid of cards that should
 * read as tactile/dimensional rather than flat. */
export function TiltCard({ children, className = "" }: { children: ReactNode; className?: string }) {
  const ref = useRef<HTMLDivElement>(null)
  const rx = useMotionValue(0)
  const ry = useMotionValue(0)
  const srx = useSpring(rx, { stiffness: 220, damping: 22 })
  const sry = useSpring(ry, { stiffness: 220, damping: 22 })

  const onMove = (e: MouseEvent<HTMLDivElement>) => {
    const rect = ref.current?.getBoundingClientRect()
    if (!rect) return
    const px = (e.clientX - rect.left) / rect.width - 0.5
    const py = (e.clientY - rect.top) / rect.height - 0.5
    ry.set(px * 7)
    rx.set(-py * 7)
  }
  const onLeave = () => { rx.set(0); ry.set(0) }

  return (
    <motion.div
      ref={ref}
      onMouseMove={onMove}
      onMouseLeave={onLeave}
      style={{ rotateX: srx, rotateY: sry, transformPerspective: 800 }}
      className={className}
    >
      {children}
    </motion.div>
  )
}

/**
 * Haikei-inspired ambient field: a couple of large, softly blurred color
 * fields that drift slowly behind hero content. CSS-driven (not a JS
 * animation loop) so it costs nothing at runtime. Decorative only -
 * aria-hidden, pointer-events-none, and used once (the Overview hero),
 * not as a global page background.
 */
export function AmbientField({ className = "" }: { className?: string }) {
  return (
    <div className={`pointer-events-none absolute overflow-hidden ${className}`} aria-hidden>
      <div className="absolute -top-20 -left-16 h-72 w-72 rounded-full bg-series-blue/16 blur-[100px] animate-[drift-a_20s_ease-in-out_infinite]" />
      <div className="absolute top-4 right-0 h-64 w-64 rounded-full bg-series-violet/12 blur-[100px] animate-[drift-b_24s_ease-in-out_infinite]" />
    </div>
  )
}
