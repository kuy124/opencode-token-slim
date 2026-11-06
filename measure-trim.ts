/**
 * Replay a real session through the plugin and report what survives the trim.
 *
 * measure.ts says where the bytes ARE; this says where they still are after
 * the rules ran, which is the only number that decides what to cut next.
 *
 * Usage: node --experimental-strip-types measure-trim.ts [sessionID]
 */
import { readFileSync } from "node:fs"
import { DatabaseSync } from "node:sqlite"
import plugin from "./index.ts"
import { dbPath, largestSession } from "./session.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")
const db = new DatabaseSync(dbPath, { readOnly: true })
const sessionID: string = process.argv[2] ?? largestSession(db)
const rows = db.prepare("SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq").all(sessionID) as {
  type: string
  data: string
}[]

const messages: any[] = []
for (const row of rows) {
  const m = JSON.parse(row.data)
  const content: any[] = []
  for (const c of m.content ?? []) {
    if (c.type === "reasoning") content.push({ type: "reasoning", text: c.text ?? "" })
    else if (c.type === "text") content.push({ type: "text", text: c.text ?? "" })
    else if (c.type === "tool") {
      const state = c.state ?? {}
      content.push({ type: "tool-call", id: c.id, name: c.name, input: state.input ?? {} })
      content.push({ type: "tool-result", id: c.id, name: c.name, result: { type: "content", value: state.content ?? [] } })
    }
  }
  messages.push({ id: m.id, content })
}

const options = JSON.parse(readFileSync("C:/Users/LENOVO/.config/opencode/opencode.json", "utf8")).plugins[1].options

const hooks: Record<string, Function> = {}
const ctx = {
  options,
  tool: { hook: async (n: string, f: Function) => void (hooks[`tool.${n}`] = f) },
  session: { hook: async (n: string, f: Function) => void (hooks[`session.${n}`] = f), synthetic: async () => {} },
  command: { transform: async (f: any) => f({ add: () => {} }) },
}
await (plugin as any).setup(ctx)

const walk = (p: any): number => {
  if (p.type === "reasoning" || p.type === "text") return bytes(p.text ?? "")
  if (p.type === "tool-call") return bytes(JSON.stringify(p.input ?? {}))
  if (p.type === "tool-result") {
    const r = p.result
    if (r?.type === "text") return bytes(r.value ?? "")
    if (r?.type === "content" && Array.isArray(r.value))
      return r.value.reduce((n: number, i: any) => n + bytes(i?.text ?? ""), 0)
  }
  return 0
}
const pay = (ms: any[]) => ms.reduce((n, m) => n + (m.content ?? []).reduce((k: number, p: any) => k + walk(p), 0), 0)

const before = pay(messages)
const tally = new Map<string, number>()
for (const m of messages) for (const p of m.content) tally.set(p.type, (tally.get(p.type) ?? 0) + walk(p))

const event = { sessionID, system: [{ type: "text", text: "s".repeat(20000) }], tools: {}, messages }
await hooks["session.context"](event)

const after = pay(event.messages)
const kind = new Map<string, number>()
const kindCount = new Map<string, number>()
for (const m of event.messages) {
  for (const p of m.content) {
    const n = walk(p)
    let k = p.type
    if (p.type === "tool-result") {
      const t = JSON.stringify(p.result)
      k = /\[slim:[^\]]*Full (?:output|text):/.test(t) ? "result/pointer" : /\[slim:/.test(t) ? "result/digest" : "result/FULL"
    }
    kind.set(k, (kind.get(k) ?? 0) + n)
    kindCount.set(k, (kindCount.get(k) ?? 0) + 1)
  }
}

const kb = (n: number) => (n / 1024).toFixed(1).padStart(9)
console.log(`session ${sessionID}: ${messages.length} messages`)
console.log(`  payload before ${kb(before)} KB`)
console.log(`  payload after  ${kb(after)} KB  (${((1 - after / before) * 100).toFixed(1)}% smaller)`)
console.log()
for (const [k, v] of [...kind].sort((a, b) => b[1] - a[1]))
  console.log(`  ${k.padEnd(16)} ${kb(v)} KB  ${String(kindCount.get(k)).padStart(5)} parts  ${((v / after) * 100).toFixed(1)}% of what remains`)
