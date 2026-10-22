/**
 * Measure a real session the way the plugin measures it.
 *
 * The plugin's request payload is the LLM message list: the session messages
 * converted by to-llm-message.ts. This script reconstructs that list from the
 * session database and reports the byte composition, so the reasoning budget
 * can be sized against real data instead of a guess.
 *
 * Usage: node --experimental-strip-types measure.ts <sessionID>
 */
import { DatabaseSync } from "node:sqlite"
import { dbPath, largestSession } from "./session.ts"

const db = new DatabaseSync(dbPath, { readOnly: true })
const sessionID = process.argv[2] ?? largestSession(db)
const rows = db
  .prepare("SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq")
  .all(sessionID) as { type: string; data: string }[]

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

type Part = { type: string; [k: string]: any }

// The LLM frame, per to-llm-message.ts: an assistant message carries its
// text, reasoning and tool-call parts; each tool call is followed by a
// tool-result part.
const parts: Part[] = []
const reasoningPerMessage: number[] = []
let signed = 0
let signedBytes = 0

for (const row of rows) {
  if (row.type !== "assistant") continue
  const m = JSON.parse(row.data)
  const content: Part[] = m.content ?? []
  let reasoningHere = 0
  for (const c of content) {
    if (c.type === "reasoning") {
      parts.push({ type: "reasoning", text: c.text ?? "" })
      reasoningHere++
      // A part that carries a provider signature has to be replayed
      // byte-identically, so it can never be pruned.
      const meta = c.providerMetadata
      const signedPart =
        typeof c.encrypted === "string" ||
        (meta && Object.values(meta).some((v: any) => v && ("signature" in v || "redactedData" in v)))
      if (signedPart) {
        signed++
        signedBytes += bytes(c.text ?? "")
      }
    } else if (c.type === "text") {
      parts.push({ type: "text", text: c.text ?? "" })
    } else if (c.type === "tool") {
      const state = c.state ?? {}
      const inputBytes = bytes(JSON.stringify(state.input ?? {}))
      const contentBytes = bytes(JSON.stringify(state.content ?? []))
      // tool-call part: the ARGUMENTS, replayed to the model every turn.
      parts.push({ type: "tool-call", id: c.id, name: c.name, input: state.input ?? {} })
      // tool-result part: the OUTPUT.
      parts.push({
        type: "tool-result",
        id: c.id,
        name: c.name,
        result: { type: "content", value: state.content ?? [] },
      })
      void inputBytes
      void contentBytes
    }
  }
  if (reasoningHere > 0) reasoningPerMessage.push(reasoningHere)
}

const size = (p: Part) => {
  if (p.type === "text" || p.type === "reasoning") return bytes(p.text ?? "")
  if (p.type === "tool-call") return bytes(JSON.stringify(p.input ?? {}))
  if (p.type === "tool-result") return bytes(JSON.stringify(p.result ?? {}))
  return bytes(JSON.stringify(p))
}

const totals = new Map<string, number>()
const counts = new Map<string, number>()
for (const p of parts) {
  totals.set(p.type, (totals.get(p.type) ?? 0) + size(p))
  counts.set(p.type, (counts.get(p.type) ?? 0) + 1)
}

const total = [...totals.values()].reduce((a, b) => a + b, 0)
console.log(`session ${sessionID}`)
console.log(`  session messages: ${rows.length}`)
console.log(`  LLM parts:        ${parts.length}`)
console.log()
for (const [k, v] of [...totals].sort((a, b) => b[1] - a[1])) {
  console.log(`  ${k.padEnd(12)} ${(v / 1024).toFixed(1).padStart(9)} KB  ${counts.get(k)} parts  ${((v / total) * 100).toFixed(1)}%`)
}
console.log(`  ${"TOTAL".padEnd(12)} ${(total / 1024).toFixed(1).padStart(9)} KB`)

const reasoningParts = parts.filter((p) => p.type === "reasoning")
const reasoningBytes = reasoningParts.reduce((n, p) => n + size(p), 0)
console.log()
console.log(`reasoning: ${reasoningParts.length} parts, ${(reasoningBytes / 1024).toFixed(1)} KB`)
const sorted = reasoningParts.map((p) => size(p)).sort((a, b) => b - a)
console.log(`  largest 10 (KB): ${sorted.slice(0, 10).map((s) => Math.round(s / 1024)).join(", ")}`)
console.log(`  median (KB):      ${(sorted[Math.floor(sorted.length / 2)] / 1024).toFixed(1)}`)
console.log(`  signed (unprunable): ${signed} parts, ${(signedBytes / 1024).toFixed(1)} KB`)

reasoningPerMessage.sort((a, b) => b - a)
console.log()
console.log(`reasoning parts per assistant message (largest 12): ${reasoningPerMessage.slice(0, 12).join(", ")}`)
console.log(`  messages with reasoning: ${reasoningPerMessage.length}`)

// How much does a byte budget of N keep?
console.log()
for (const budget of [16_000, 32_000, 64_000, 128_000, 256_000]) {
  let acc = 0
  let kept = 0
  for (let i = reasoningParts.length - 1; i >= 0; i--) {
    const b = size(reasoningParts[i])
    if (acc + b > budget) break
    acc += b
    kept++
  }
  console.log(
    `  budget ${String(budget / 1024).padStart(4)} KB -> keeps ${String(kept).padStart(4)} parts, ${(acc / 1024).toFixed(1).padStart(8)} KB  (saves ${((reasoningBytes - acc) / 1024).toFixed(1)} KB)`,
  )
}
