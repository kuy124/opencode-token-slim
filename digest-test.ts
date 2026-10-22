/**
 * Regression test: a digest must keep the head and the tail as CONTIGUOUS
 * line ranges and must name the omitted range.
 *
 * The bug: the digest spliced the head and the tail together with a bare
 * "..." between them, so a file that jumped from line 10 to line 718 looked
 * like it had a corrupt region. A model reading that went hunting for a
 * "broken line" and burned 4.6x the tokens re-checking the file.
 *
 * This test drives the REAL digestResult from index.ts, not a copy, so it
 * cannot drift away from the shipped code.
 */
import assert from "node:assert/strict"
import { digestResult, type SlimOptions } from "./index.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

const OPTS: SlimOptions = {
  resultBytes: 12000,
  headBytes: 3000,
  tailBytes: 1500,
  reasoningBytes: 80000,
  reasoningKeepBytes: 32000,
  stairBytes: 16000,
  resultKeepBytes: 32000,
  resultDigestBytes: 48000,
  callKeepBytes: 8000,
  textKeepBytes: 16000,
  elideFloorBytes: 512,
  systemBytes: 24000,
  toolBytes: 10000,
  duplicateBytes: 2000,
  stripItemIds: [],
  keepFullOutput: false,
  directory: "",
  enabled: true,
  snip: true,
  snipPath: "",
  snipTimeoutMs: 2000,
  protectPaths: [],
  protectTools: [],
  maxDiskBytes: 256 * 1024 * 1024,
  statsDirectory: "",
}

const digest = (text: string) => digestResult(text, OPTS, "read")

// A read result shaped like the one that broke: numbered lines.
const lines = Array.from({ length: 900 }, (_, i) => `${i + 1}: export const item_${i} = { id: ${i} }`)
const text = lines.join("\n")

const out = digest(text).split("\n")

// 1. The head must be contiguous and start at line 1.
assert.ok(out[0].startsWith("1: "), "head starts at line 1")
const headCount = out.findIndex((l) => l.startsWith("[slim:"))
assert.ok(headCount > 1, "head has content")
for (let i = 0; i < headCount; i++) {
  assert.ok(out[i].startsWith(`${i + 1}: `), `head line ${i + 1} is contiguous, got: ${out[i].slice(0, 30)}`)
}

// 2. The marker must name the omitted range explicitly.
const marker = out[headCount]
assert.ok(marker.includes("[slim:"), "marker present")
assert.ok(/Lines \d+-\d+ \(\d+ lines\) omitted/.test(marker), `marker names the range: ${marker}`)
const m = marker.match(/Lines (\d+)-(\d+) \((\d+) lines\)/)
assert.ok(m, "range parses")
const from = Number(m[1])
const to = Number(m[2])
assert.equal(to - from + 1, Number(m[3]), "range length is consistent")
assert.equal(from, headCount + 1, "omitted range starts right after the head")

// 3. The tail must be contiguous and pick up exactly where the range ends.
const tail = out.slice(headCount + 1)
assert.ok(tail.length > 0, "tail present")
assert.ok(tail[0].startsWith(`${to + 1}: `), `tail starts at line ${to + 1}, got: ${tail[0].slice(0, 30)}`)
for (let i = 0; i < tail.length; i++) {
  assert.ok(tail[i].startsWith(`${to + 1 + i}: `), `tail line ${i} is contiguous`)
}

// 4. The digest must be smaller than the original and respect the budget.
assert.ok(bytes(digest(text)) < bytes(text), "digest is smaller")
assert.ok(bytes(digest(text)) <= 3000 + 1500 + 300, "digest respects the budget")

// 5. A result that fits must not claim anything was omitted.
const small = digest("hello\nworld")
assert.ok(!small.includes("omitted"), "a small result claims no omission")
assert.ok(small.includes("hello"), "content preserved")

// 6. The marker must not look like a broken file.
assert.ok(!/^\.\.\.$/m.test(digest(text)), "no bare ellipsis line that reads as corruption")

// 7. A single-line result must not produce an empty head or tail.
const single = digest("just one line")
assert.ok(single.includes("just one line"), "single-line content survives")
assert.ok(!single.includes("Lines "), "single-line result claims no omission")

// 8. A result just over the head budget must keep its first line.
const tight = digestResult(lines.slice(0, 40).join("\n"), { ...OPTS, headBytes: 20, tailBytes: 20 }, "read")
assert.ok(tight.startsWith("1: "), "tight digest keeps the first line")
assert.ok(/Lines \d+-\d+ \(\d+ lines\) omitted/.test(tight), "tight digest names the range")

// 9. A result whose single line busts the budget must NOT be digested to
// nothing. The old code dropped the head and the tail entirely and left a
// bare marker, so the model got zero content.
const oneBig = "A".repeat(50_000)
const d1 = digest(oneBig)
assert.ok(d1.includes("AAAA"), "single oversized line keeps content")
assert.ok(d1.includes("truncated to fit the budget"), "clipping is stated")
assert.ok(bytes(d1) < bytes(oneBig), "clipped digest is still smaller")

// 10. Two oversized lines: the first and the last must both survive.
const twoBig = "B".repeat(20_000) + "\n" + "C".repeat(20_000)
const d2 = digest(twoBig)
assert.ok(d2.includes("BBBB"), "first oversized line survives")
assert.ok(d2.includes("CCCC"), "last oversized line survives")

// 11. A short result must keep its last line even though nothing was omitted.
const twoSmall = "first line\nsecond line"
const d3 = digest(twoSmall)
assert.ok(d3.includes("first line"), "short result keeps line 1")
assert.ok(d3.includes("second line"), "short result keeps line 2")
assert.ok(!d3.includes("omitted"), "short result claims no omission")

// 12. The digest must never be larger than the original.
for (const sample of [oneBig, twoBig, twoSmall, text, lines.slice(0, 3).join("\n")]) {
  assert.ok(bytes(digest(sample)) <= bytes(sample), "digest never exceeds the original")
}

console.log("all checks passed")
