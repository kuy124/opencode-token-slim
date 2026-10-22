/**
 * Is the reduction CONSISTENT?
 *
 * The complaint that started this was that the saving swung: 5% on one
 * turn, 49% on the next, 15% over a session. A rule expressed in message
 * counts cannot be consistent, because the content of a session is uneven.
 *
 * This replays a real session as a series of growing prefixes — a session
 * of 100 messages, then 200, then 300 — and prints the reduction at each
 * step. A consistent rule holds the reduction roughly steady as the session
 * grows, instead of swinging with the shape of the content.
 */
import { DatabaseSync } from "node:sqlite"
import plugin from "./index.ts"
import { dbPath, largestSession } from "./session.ts"

const db = new DatabaseSync(dbPath, { readOnly: true })
const sessionID = process.argv[2] ?? largestSession(db)
const rows = db
  .prepare("SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq")
  .all(sessionID) as { type: string; data: string }[]

const bytes = (s: string) => Buffer.byteLength(s, "utf8")
const partBytes = (part: any): number => {
  if (!part) return 0
  if (part.type === "text" || part.type === "reasoning") return bytes(part.text ?? "")
  if (part.type === "tool-result") return bytes(JSON.stringify(part.result ?? {}))
  if (part.type === "tool-call") return bytes(JSON.stringify(part.input ?? {}))
  return bytes(JSON.stringify(part))
}

// The full LLM message list, once.
const all: any[] = []
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
  if (content.length) all.push({ id: `m${all.length}`, role: "assistant", content })
}

const system = [
  { type: "text", text: "# Code Mode\n" + "x".repeat(37_000) },
  { type: "text", text: "y".repeat(1_800) },
]
const tools = {
  read: { description: "d".repeat(3_000), input: { type: "object", properties: { path: { type: "string", description: "p".repeat(2_000) } } } },
  bash: { description: "b".repeat(3_000), input: { type: "object", properties: { cmd: { type: "string", description: "q".repeat(2_000) } } } },
}

const payloadOf = (msgs: any[]) =>
  system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
  Object.values(tools as any).reduce(
    (n: number, t: any) => n + bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {})),
    0,
  ) +
  msgs.reduce(
    (n: number, m: any) => n + (m.content ?? []).reduce((k: number, p: any) => k + partBytes(p), 0),
    0,
  )

function mockCtx(options: any) {
  const hooks: Record<string, Function> = {}
  const ctx = {
    options,
    tool: { hook: async (n: string, f: Function) => { hooks[`tool.${n}`] = f } },
    session: {
      hook: async (n: string, f: Function) => { hooks[`session.${n}`] = f },
      synthetic: async () => {},
    },
    command: { transform: async (f: Function) => { f({ add: () => {} }) } },
  }
  return { ctx, hooks }
}

const OPTIONS = JSON.parse(process.env.SLIM_OPTIONS ?? "{}")

// The full session, measured once, and then growing prefixes.
console.log("session length   payload before   payload after   reduction")
console.log("-------------   --------------   -------------   ---------")

// A single plugin instance, driven repeatedly, the way a live session is:
// the ledger accumulates and the staircase advances.
//
// keepFullOutput defaults to true, the same as the plugin: with it off the
// plugin cannot write the old results to disk, so it has to digest them
// instead of pointing at them, and the saving is much smaller. That is a
// real difference and it is measured separately, not hidden.
const { ctx, hooks } = mockCtx({ keepFullOutput: true, ...OPTIONS })
await (plugin as any).setup(ctx)

const steps = [20, 40, 60, 100, 150, 200, 260, 320, 380, 440, all.length]
for (const n of steps) {
  if (n > all.length) continue
  const event = {
    sessionID: "consistency",
    model: { providerID: "ollama", id: "minimax-m3.1-flash" },
    system: system.map((p) => ({ ...p })),
    tools: JSON.parse(JSON.stringify(tools)),
    messages: all.slice(0, n).map((m) => ({ ...m, content: [...m.content] })),
  }
  const before = payloadOf(event.messages)
  await hooks["session.context"](event)
  const after = payloadOf(event.messages)
  const pct = ((1 - after / before) * 100).toFixed(1)
  console.log(
    `${String(n).padStart(13)}   ${(before / 1024).toFixed(0).padStart(12)} KB   ${(after / 1024).toFixed(0).padStart(11)} KB   ${pct.padStart(8)}%`,
  )
}
