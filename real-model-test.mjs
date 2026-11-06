import { spawnSync } from "node:child_process"

const model = process.env.REAL_MODEL || "ollama-cloud/beam"

const proc = spawnSync(
  "opencode",
  ["run", "-m", model, "--auto", "Respond with exactly one word: OK", "--format", "json"],
  {
    encoding: "utf8",
    timeout: 290_000,
    maxBuffer: 50 * 1024 * 1024,
    shell: true,
  },
)

const out = `${proc.stdout || ""}${proc.stderr || ""}`
console.log(out.slice(-500))

if (proc.error) {
  console.error("spawn failed:", proc.error)
  process.exit(1)
}
if (proc.status !== 0) {
  console.error("MODEL TEST EXIT NON-ZERO:", proc.status)
  console.error(out.slice(-2_000))
  process.exit(1)
}
if (!/OK/i.test(out)) {
  console.error("MODEL TEST DID NOT RESPOND OK")
  process.exit(1)
}
console.log("REAL MODEL TEST PASSED:", model)
