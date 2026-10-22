/**
 * Where the tool-result bytes are, and what a digest of each would cost.
 *
 * The point: a digest has to be smaller than the result it replaces, and
 * the per-result cost of the marker adds up over hundreds of results.
 */
import { DatabaseSync } from "node:sqlite"
import { dbPath, largestSession } from "./session.ts"

const db = new DatabaseSync(dbPath, { readOnly: true })
const sessionID = process.argv[2] ?? largestSession(db)
const rows = db
  .prepare("SELECT type, data FROM session_message WHERE session_id = ? ORDER BY seq")
  .all(sessionID) as { type: string; data: string }[]

const sizes: number[] = []
for (const r of rows) {
  if (r.type !== "assistant") continue
  const m = JSON.parse(r.data)
  for (const p of m.content ?? []) {
    if (p.type !== "tool") continue
    const c = p.state?.content ?? []
    const t = c.filter((x: any) => x.type === "text").map((x: any) => x.text).join("\n")
    if (t) sizes.push(Buffer.byteLength(t, "utf8"))
  }
}
sizes.sort((a, b) => b - a)

const total = sizes.reduce((a, b) => a + b, 0)
console.log(`results: ${sizes.length}   total ${(total / 1024).toFixed(0)} KB`)

for (const floor of [0, 1000, 2000, 3000, 5000]) {
  const over = sizes.filter((s) => s > floor)
  const bytes = over.reduce((a, b) => a + b, 0)
  // digest cost with the proportional budget
  let dig = 0
  for (const s of over) {
    const cap = Math.floor(s / 3)
    dig += Math.min(3000, cap) + Math.min(1500, Math.floor(cap / 2)) + 150
  }
  const kept = total - bytes
  console.log(
    `  floor ${String(floor).padStart(5)}: ${String(over.length).padStart(4)} digested, ` +
      `orig ${(bytes / 1024).toFixed(0).padStart(5)} KB -> digest ${(dig / 1024).toFixed(0).padStart(5)} KB, ` +
      `kept ${(kept / 1024).toFixed(0).padStart(5)} KB, net ${(((total - kept - dig) / total) * 100).toFixed(0)}% cut`,
  )
}
