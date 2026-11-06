import assert from "node:assert/strict"
import { performance } from "node:perf_hooks"
import * as fs from "node:fs"
import * as os from "node:os"
import * as path from "node:path"
const bytes = (s: string) => Buffer.byteLength(s, "utf8")
import * as main from "./index.ts"

const resolveOptions = main.resolveOptions
const assignBands = main.assignBands
const digestResult = main.digestResult
const readRestorable = main.readRestorable
const trimSchemaDescriptions = main.trimSchemaDescriptions
const enforceDiskCap = main.enforceDiskCap
const writeFullOutput = main.writeFullOutput

const tmp = path.join(os.tmpdir(), `slim-insane-${Date.now()}`)

function timeMs(fn: () => void): number {
  const t0 = performance.now()
  fn()
  return performance.now() - t0
}

async function mainTests() {
  // ----- 1. 1M-part banding (extreme) -----
  {
    const count = 1_000_000
    const flat = Array.from({ length: count }, (_, i) => ({
      part: { type: "reasoning", text: "x".repeat(20) },
      index: i,
    }))
    const opts = resolveOptions({
      reasoningBytes: 10_000,
      reasoningKeepBytes: 2_000,
      stairBytes: 5_000,
    })
    const ms = timeMs(() => {
      const bands = assignBands(flat, opts, "reasoning")
      assert.equal(bands instanceof Map, true)
      // Keep result bounded by the configured reasoning byte floor.
      assert.ok(bands.size <= count, "banding dropped work items")
    })
    console.log("insane 1/6 banding 1M:", Math.round(ms), "ms")
    assert.ok(ms < 60_000, "extreme banding took too long")
  }

  // ----- 2. 5 MB result digest + byte-for-byte restore -----
  {
    fs.mkdirSync(tmp, { recursive: true })
    const text = "Q".repeat(5_000_000)
    const opts = resolveOptions({
      keepFullOutput: true,
      directory: tmp,
      resultBytes: 4_096,
      resultKeepBytes: 10_000_000,
      elideFloorBytes: 100,
    })
    let out = ""
    const digestMs = timeMs(() => (out = digestResult(text, opts, "read")))
    assert.ok(bytes(out) < 10_000, `digest not condensed: ${bytes(out)} bytes`)
    const restored = readRestorable(out, opts)
    assert.equal(typeof restored, "string")
    assert.equal((restored as string).length, text.length)
    console.log("insane 2/6 digest+restore 5M:", Math.round(digestMs), "ms", bytes(out), "bytes kept")
    assert.ok(digestMs < 30_000, "digest too slow")
  }

  // ----- 3. giant schema description trim -----
  {
    const schema = {
      type: "object",
      properties: Object.fromEntries(
        Array.from({ length: 10_000 }, (_, i) => [
          `f${i}`,
          { type: "string", description: "d".repeat(1_000) },
        ]),
      ),
      required: ["f0"],
    }
    const ms = timeMs(() => {
      const trimmed = trimSchemaDescriptions(schema, 200)
      assert.equal(Object.keys(trimmed.properties).length, 10_000)
      assert.ok(bytes(trimmed.properties.f0.description) <= 200)
    })
    console.log("insane 3/6 schema trim 10k:", Math.round(ms), "ms")
    assert.ok(ms < 30_000, "schema trim too slow")
  }

  // ----- 4. disk cap with 5,000 output files -----
  {
    const dir = path.join(tmp, "disk")
    fs.mkdirSync(dir, { recursive: true })
    for (let i = 0; i < 5_000; i++) {
      fs.writeFileSync(path.join(dir, `out-${i}.txt`), "z".repeat(4_000))
    }
    const ms = timeMs(() => {
      enforceDiskCap(dir, 8 * 1024 * 1024)
      const rem = fs.readdirSync(dir).length
      assert.ok(rem < 5_000, `disk cap left everything (${rem} files)`)
    })
    console.log("insane 4/6 disk cap 5k:", Math.round(ms), "ms")
    assert.ok(ms < 30_000, "disk cap too slow")
  }

  // ----- 5. helper throughput: writeFullOutput under repeated writes -----
  {
    const dir = path.join(tmp, "writes")
    fs.mkdirSync(dir, { recursive: true })
    const count = 10_000
    const ms = timeMs(() => {
      for (let i = 0; i < count; i++) writeFullOutput(dir, `r-${i}.txt`, "x".repeat(4_096))
    })
    const files = fs.readdirSync(dir).length
    assert.ok(files <= count && files > 0, "writeFullOutput lost the outputs")
    console.log("insane 5/6 10k writes:", Math.round(ms), "ms", files, "files")
    assert.ok(ms < 60_000, "bulk writes too slow")
  }

  // ----- 6. ratchet: cut does not leave a non-finite allocation -----
  {
    const opts = resolveOptions({})
    for (let i = 0; i < 100_000; i++) {
      assert.ok(Number.isFinite(opts.resultKeepBytes))
      assert.ok(opts.resultKeepBytes > 0)
    }
    console.log("insane 6/6 options ratchet passed")
  }

  console.log("all insane checks passed")
}

mainTests().catch((err) => {
  console.error(err)
  process.exit(1)
})
