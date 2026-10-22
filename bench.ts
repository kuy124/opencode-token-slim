/**
 * The plugin's own pruning, run against a real session.
 *
 * The measurement in the README came from a synthetic session. This runs the
 * REAL context hook over a session rebuilt from the database, so the before
 * and after numbers come from the same code that runs in production.
 *
 * Usage: node --experimental-strip-types bench.ts [sessionID]
 */
import { DatabaseSync } from "node:sqlite"
import plugin from "./index.ts"
import { dbPath, largestSession } from "./session.ts"

const db = new DatabaseSync(dbPath, { readOnly: true })
// Flags (`--curve`) are not session ids: only the bare arguments are.
const sessionID = process.argv.slice(2).find((a) => !a.startsWith("--")) ?? largestSession(db)

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

const rows = db
  .prepare("SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq")
  .all(sessionID) as { type: string; data: string }[]

// Rebuild the LLM message list, per to-llm-message.ts: an assistant message
// carries its text, reasoning and tool-call parts, and each tool call is
// followed by a tool-result part.
const messages: any[] = []
for (const row of rows) {
  if (row.type !== "assistant") continue
  const m = JSON.parse(row.data)
  const content: any[] = []
  for (const c of m.content ?? []) {
    if (c.type === "reasoning") content.push({ type: "reasoning", text: c.text ?? "" })
    else if (c.type === "text") content.push({ type: "text", text: c.text ?? "" })
    else if (c.type === "tool") {
      content.push({ type: "tool-call", id: c.id, name: c.name, input: c.state?.input ?? {} })
      content.push({
        type: "tool-result",
        id: c.id,
        name: c.name,
        result: { type: "content", value: c.state?.content ?? [] },
      })
    }
  }
  if (content.length) messages.push({ id: `m${messages.length}`, role: "assistant", content })
}

const system = [
  { type: "text", text: "# Code Mode\n" + "x".repeat(37_000) },
  { type: "text", text: "y".repeat(1_800) },
]
const tools = {
  read: { description: "d".repeat(3_000), input: { type: "object", properties: { path: { type: "string", description: "p".repeat(2_000) } } } },
  bash: { description: "b".repeat(3_000), input: { type: "object", properties: { cmd: { type: "string", description: "q".repeat(2_000) } } } },
}

const partBytes = (part: any): number => {
  if (!part) return 0
  if (part.type === "text" || part.type === "reasoning") return bytes(part.text ?? "")
  if (part.type === "tool-result") return bytes(JSON.stringify(part.result ?? {}))
  if (part.type === "tool-call") return bytes(JSON.stringify(part.input ?? {}))
  return bytes(JSON.stringify(part))
}
const cat = (msgs: any[]) => {
  const out: Record<string, number> = {}
  for (const m of msgs) {
    for (const p of m.content ?? []) {
      const k = p.type
      out[k] = (out[k] ?? 0) + partBytes(p)
    }
  }
  return out
}
const payloadOf = (e: any) =>
  (e.system ?? []).reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
  Object.values(e.tools ?? {}).reduce(
    (n: number, t: any) => n + bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {})),
    0,
  ) +
  (e.messages ?? []).reduce(
    (n: number, m: any) => n + (m.content ?? []).reduce((k: number, p: any) => k + partBytes(p), 0),
    0,
  )

function mockCtx(options: any) {
  const hooks: Record<string, Function> = {}
  const synthetic: any[] = []
  const commands: any[] = []
  const ctx = {
    options,
    tool: { hook: async (n: string, f: Function) => { hooks[`tool.${n}`] = f } },
    session: {
      hook: async (n: string, f: Function) => { hooks[`session.${n}`] = f },
      synthetic: async (e: any) => { synthetic.push(e) },
    },
    command: { transform: async (f: Function) => { f({ add: (c: any) => commands.push(c) }) } },
  }
  return { ctx, hooks, synthetic, commands }
}

const OPTIONS = JSON.parse(process.env.SLIM_OPTIONS ?? "{}")
// keepFullOutput defaults to true, the same as the plugin: with it off the
// old results cannot be pointed at on disk, so they have to be digested
// instead, and the saving is much smaller. Set SLIM_OPTIONS to measure the
// other case.
const { ctx, hooks } = mockCtx({ keepFullOutput: true, ...OPTIONS })
await (plugin as any).setup(ctx)

const event = {
  sessionID: "bench",
  model: { providerID: "ollama", id: "minimax-m3.1-flash" },
  system: system.map((p) => ({ ...p })),
  tools: JSON.parse(JSON.stringify(tools)),
  messages: messages.map((m) => ({ ...m, content: [...m.content] })),
}

const before = payloadOf(event)
// Snapshot the categories BEFORE the hook, from a copy.
const catB = cat(event.messages.map((m) => ({ ...m, content: [...m.content] })))
await hooks["session.context"](event)
const after = payloadOf(event)
const catA = cat(event.messages)

const cats: Record<string, { b: number; a: number }> = {}
for (const k of new Set([...Object.keys(catB), ...Object.keys(catA)])) {
  cats[k] = { b: catB[k] ?? 0, a: catA[k] ?? 0 }
}

const pct = ((1 - after / before) * 100).toFixed(1)
console.log(`session ${sessionID}   messages: ${messages.length}`)
console.log(`  options: ${JSON.stringify(OPTIONS)}`)
console.log()
console.log(`  BEFORE  ${(before / 1024).toFixed(1).padStart(9)} KB`)
console.log(`  AFTER   ${(after / 1024).toFixed(1).padStart(9)} KB   ${pct}% smaller`)
console.log()
for (const [k, v] of Object.entries(cats)) {
  const cut = v.b - v.a
  console.log(
    `  ${k.padEnd(12)} ${(v.b / 1024).toFixed(1).padStart(8)} KB -> ${(v.a / 1024).toFixed(1).padStart(8)} KB   cut ${(cut / 1024).toFixed(1).padStart(8)} KB  ${v.b ? (((cut / v.b) * 100).toFixed(0) + "%").padStart(5) : ""}`,
  )
}
console.log()
const sysB = system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
  Object.values(tools as any).reduce((n: number, t: any) => n + bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {})), 0)
const sysA = event.system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
  Object.values(event.tools as any).reduce((n: number, t: any) => n + bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {})), 0)
console.log(`  system+tools ${(sysB / 1024).toFixed(1).padStart(8)} KB -> ${(sysA / 1024).toFixed(1).padStart(8)} KB`)

// --curve: the same hook over every prefix of the history. The single number
// above is the LAST request; this shows whether the trimming happens on every
// request or only once the session is already long.
if (process.argv.includes("--curve")) {
  console.log()
  console.log("  per request (prefix replayed through the same hook)")
  const cuts: number[] = []
  for (let i = 25; i <= messages.length; i += 25) cuts.push(i)
  if (cuts.at(-1) !== messages.length) cuts.push(messages.length)
  for (const cut of cuts) {
    const e = {
      sessionID: "bench",
      model: { providerID: "ollama", id: "minimax-m3.1-flash" },
      system: system.map((p) => ({ ...p })),
      tools: JSON.parse(JSON.stringify(tools)),
      messages: structuredClone(messages.slice(0, cut)),
    }
    const b = payloadOf(e)
    await hooks["session.context"](e)
    const a = payloadOf(e)
    const p = (((1 - a / b) * 100).toFixed(0) + "%").padStart(5)
    console.log(
      `  ${String(cut).padStart(5)} msgs ${(b / 1024).toFixed(1).padStart(9)} KB -> ${(a / 1024).toFixed(1).padStart(9)} KB ${p}`,
    )
  }
}