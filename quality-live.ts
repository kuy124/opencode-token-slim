/**
 * Does the revamped plugin cost quality?
 *
 * Runs a set of tasks that are sensitive to what the plugin trims, with the
 * plugin ON and OFF, and checks every answer against a value computed here.
 * The point is not the tokens: it is that the answers do not change.
 *
 * Usage: node --experimental-strip-types quality-live.ts [--runs 1]
 */
import { execFileSync, spawn } from "node:child_process"
import { mkdtempSync, writeFileSync, rmSync, readFileSync, copyFileSync, existsSync } from "node:fs"
import { tmpdir } from "node:os"
import { join } from "node:path"

const sleep = (ms: number) => new Promise((r) => setTimeout(r, ms))
const runsFlag = process.argv.indexOf("--runs")
const RUNS = runsFlag > 0 ? Number(process.argv[runsFlag + 1]) || 1 : 1
const MODEL = "ollama/minimax-m3.1-flash"
const CONFIG = join(process.env.USERPROFILE, ".config", "opencode", "opencode.json")
const CONFIG_BACKUP = CONFIG + ".qualitybak"

// --- the fixture ------------------------------------------------------------
const LINES = 900
const COL = []
const rows: string[] = []
for (let i = 1; i <= LINES; i++) {
  const v = (i * 7) % 97
  COL.push(v)
  rows.push(`line ${i}: value=${v} label=${i % 2 === 0 ? "even" : "odd"}`)
}
const NEEDLE_LINE = 617
const NEEDLE_VALUE = COL[NEEDLE_LINE - 1]
// A bounded sum. Asking a small model to add 900 numbers is a test of the
// model, not of the plugin, and it does not finish.
const SUM_FROM = 1
const SUM_TO = 20
const SUM_TOTAL = COL.slice(SUM_FROM - 1, SUM_TO).reduce((a, b) => a + b, 0)

const dir = mkdtempSync(join(tmpdir(), "slim-quality-"))
writeFileSync(join(dir, "data.txt"), rows.join("\n"))
writeFileSync(join(dir, "left.txt"), rows.slice(0, 450).join("\n"))
writeFileSync(join(dir, "right.txt"), rows.slice(450).join("\n"))

type Task = { name: string; prompt: string; check: (out: string) => boolean }
const TASKS: Task[] = [
  {
    name: "needle",
    prompt: `Read data.txt in ${dir} and tell me ONLY the value= on line ${NEEDLE_LINE}. Answer with just the number.`,
    check: (o) => new RegExp(`\\b${NEEDLE_VALUE}\\b`).test(o),
  },
  {
    name: "count",
    prompt: `Read data.txt in ${dir} and tell me how many lines it has. Answer with just the number.`,
    check: (o) => new RegExp(`\\b${LINES}\\b`).test(o),
  },
  {
    name: "sum",
    prompt: `Read data.txt in ${dir}. What is the sum of the value= numbers on lines ${SUM_FROM} to ${SUM_TO}? Answer with just the number.`,
    check: (o) => new RegExp(`\\b${SUM_TOTAL}\\b`).test(o),
  },
  {
    name: "cite",
    prompt: `Read data.txt in ${dir} and tell me which line has value=${NEEDLE_VALUE}. Answer with just the line number.`,
    check: (o) => new RegExp(`\\b${NEEDLE_LINE}\\b`).test(o),
  },
  {
    name: "cross-file",
    prompt: `Read left.txt and right.txt in ${dir}. They are two halves of one file. How many lines do they have together? Answer with just the number.`,
    check: (o) => new RegExp(`\\b${LINES}\\b`).test(o),
  },
  {
    name: "edit-then-reason",
    prompt: `Read data.txt in ${dir}. Then use the edit tool to change the line "line ${NEEDLE_LINE}: value=${NEEDLE_VALUE} label=${
      NEEDLE_LINE % 2 === 0 ? "even" : "odd"
    }" so that it says "line ${NEEDLE_LINE}: value=${NEEDLE_VALUE} label=patched". Then read the file again and tell me the label on line ${NEEDLE_LINE}. Answer with just the label.`,
    check: (o) => /patched/i.test(o),
  },
]

// --- the run ----------------------------------------------------------------
/**
 * Run one task and return its output.
 *
 * The output goes to a file, not a pipe. A pipe stays open as long as ANY
 * process holds it, and `opencode` spawns children, so a timed-out run would
 * keep the harness waiting forever on a pipe nobody is going to close. With
 * a file, the timeout can kill the process tree and move on.
 */
async function opencode(prompt: string, dir: string, timeoutMs: number): Promise<string> {
  const out = join(dir, "out.txt")
  const done = join(dir, "done.txt")
  const scriptFile = join(dir, "run.ps1")
  rmSync(out, { force: true }) // the last run's output must not be mistaken for this one
  rmSync(done, { force: true })
  const ps = (s: string) => "'" + s.replace(/'/g, "''") + "'"
  // The script goes in a .ps1 file and is run with -File. Passing it as an
  // argument means the Windows command line gets a say in what the prompt
  // looks like, and the prompt is full of quotes.
  //
  // Out-File with -Encoding utf8, not the `*>` redirect: the redirect writes
  // UTF-16LE, and a file that is read as UTF-8 is a file of mojibake.
  //
  // The sentinel is a SEPARATE file. Out-File holds the file it is writing
  // open until the pipeline ends, so the sentinel cannot be appended to the
  // same file.
  const script = [
    `$env:Path = $env:USERPROFILE + '\\go\\bin;' + $env:Path`,
    `& opencode.cmd run ${ps(prompt)} --model ${ps(MODEL)} 2>&1 | Out-File -FilePath ${ps(out)} -Encoding utf8`,
    `Set-Content -Path ${ps(done)} -Value 'DONE' -NoNewline -Encoding utf8`,
  ].join("; ")
  writeFileSync(scriptFile, script, "utf8")
  // NOT detached. On Windows, a detached powershell.exe exits immediately
  // without running the script at all, and there is no error to explain it.
  // The process tree is still killed with taskkill /T when the time is up.
  const child = spawn(
    "powershell.exe",
    ["-NoProfile", "-ExecutionPolicy", "Bypass", "-File", scriptFile],
    { cwd: dir, stdio: "ignore" },
  )
  child.on("error", (e) => console.log("  spawn error:", e.message))
  const start = Date.now()
  let finished = false
  while (Date.now() - start < timeoutMs) {
    if (existsSync(done)) {
      finished = true
      break
    }
    await sleep(1000)
  }
  if (!finished) console.log("  TIMED OUT after", Math.round((Date.now() - start) / 1000), "s")
  try {
    // The whole tree, not just powershell: opencode is its child.
    execFileSync("taskkill", ["/PID", String(child.pid), "/T", "/F"], { stdio: "ignore" })
  } catch {
    // already gone
  }
  if (!finished) return ""
  try {
    return readFileSync(out, "utf8")
  } catch {
    return ""
  }
}

function setPlugin(on: boolean) {
  const cfg = JSON.parse(readFileSync(CONFIG, "utf8"))
  const plugins = (cfg.plugins ?? []).filter(
    (p: any) => typeof p !== "object" || !String(p.package ?? "").includes("opencode-token-slim"),
  )
  if (on) {
    plugins.push({
      package: "F:\\GAMES\\Code\\opencode-token-slim",
      options: { keepFullOutput: true },
    })
  }
  cfg.plugins = plugins
  writeFileSync(CONFIG, JSON.stringify(cfg, null, 2))
}

// The user's real config is edited to toggle the plugin, and restored from a
// byte-for-byte copy at the end. Leaving it half-rewritten would be a nasty
// surprise.
copyFileSync(CONFIG, CONFIG_BACKUP)

const results: { task: string; on: boolean; pass: boolean }[] = []
for (const on of [false, true]) {
  setPlugin(on)
  for (let r = 0; r < RUNS; r++) {
    for (const t of TASKS) {
      let out = ""
      try {
        out = await opencode(t.prompt, dir, 180_000)
      } catch (e: any) {
        out = `ERROR: ${String(e.message).slice(0, 200)}`
      }
      const pass = t.check(out)
      results.push({ task: t.name, on, pass })
      console.log(`  ${on ? "ON " : "off"}  ${t.name.padEnd(16)} ${pass ? "pass" : "FAIL"}`)
      if (!pass) console.log(`        out: ${out.slice(0, 400).replace(/\n/g, "\\n")}`)
    }
  }
}

setPlugin(true)
copyFileSync(CONFIG_BACKUP, CONFIG)
rmSync(CONFIG_BACKUP, { force: true })
rmSync(dir, { recursive: true, force: true })

const off = results.filter((x) => !x.on)
const on = results.filter((x) => x.on)
console.log(`\nplugin off: ${off.filter((x) => x.pass).length}/${off.length} pass`)
console.log(`plugin on:  ${on.filter((x) => x.pass).length}/${on.length} pass`)
const changed = off.length === on.length && off.every((x, i) => x.pass === on[i].pass)
console.log(changed ? "the answers are identical" : "AN ANSWER CHANGED")
process.exit(changed ? 0 : 1)
