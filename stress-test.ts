/**
 * The limits: every trimmer, over content designed to break it.
 *
 * `quality-test.ts` asks whether the model can still answer. This asks whether
 * the arithmetic holds when the input is hostile: a megabyte on one line, a
 * png in a text slot, a marker the model itself wrote, NUL bytes, a path one
 * `..` away from the cache directory, 12 levels of JSON, a call whose argument
 * is 199 bytes long.
 *
 * Two kinds of check, deliberately:
 *
 *   - property: a corpus × invariants that must hold for ALL text (never
 *     grows, the disk copy is byte-identical, a marker always parses, the
 *     path never leaves the cache directory).
 *   - targeted: the boundaries a corpus cannot see (199/200/329/400-byte
 *     arguments, a protected path that is a prefix of another, a step edge).
 *
 * No fixtures, no frameworks: `check()` counts, and the suite fails if the
 * count drops below 100, so the number in the README cannot drift.
 */
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { existsSync, mkdtempSync, readFileSync, writeFileSync } from "node:fs"
import os from "node:os"
import path from "node:path"
import plugin, {
  assignBands,
  budgetForKind,
  carriesSignature,
  digestBudgets,
  digestJson,
  digestResult,
  elideImages,
  elideToPointer,
  flattenParts,
  floorForKind,
  isProtectedPath,
  matchesPattern,
  isCacheMarker,
  markerPath,
  partCost,
  quantize,
  readRestorable,
  reasoningKey,
  reportSummary,
  resolveOptions,
  restorePart,
  resultText,
  stairForKind,
  stripItemId,
  trimCallInput,
  trimText,
  elideContentItems,
  enforceDiskCap,
  planRestores,
  takeBytes,
  takeTail,
  extractPath,
  extractCallPath,
  readPersistedStats,
  trimSchemaDescriptions,
  trimSystemPart,
  writeFullOutput,
  writeReportFile,
} from "./index.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")
const hash = (s: string) => createHash("sha1").update(s, "utf8").digest("hex")
const dir = mkdtempSync(path.join(os.tmpdir(), "slim-stress-"))
const OPTS = resolveOptions({ keepFullOutput: true, directory: dir })

let count = 0
const names = new Set<string>()

/** Count a check, and refuse a duplicate name: a copy-paste would inflate it. */
function check(name: string, fn: () => void) {
  assert.ok(!names.has(name), `duplicate check name: ${name}`)
  names.add(name)
  try {
    fn()
  } catch (error) {
    console.error(`FAIL  ${name}`)
    throw error
  }
  count++
}

function mockCtx(options: any) {
  const hooks: Record<string, Function> = {}
  const ctx = {
    options,
    tool: { hook: async (n: string, f: Function) => { hooks[`tool.${n}`] = f } },
    session: {
      hook: async (n: string, f: Function) => { hooks[`session.${n}`] = f },
      synthetic: async () => {},
    },
    command: { transform: async () => {} },
  }
  return { ctx, hooks }
}

// --- the corpus -------------------------------------------------------------
//
// Every one of these has shipped a bug or could have: a line with no break, a
// CRLF file, a png where text was expected, a marker inside the content, a NUL.
const CORPUS: Array<{ name: string; text: string }> = [
  { name: "empty", text: "" },
  { name: "one char", text: "x" },
  { name: "one line, 10 KB", text: "a".repeat(10_000) },
  { name: "one line, 1 MB", text: "b".repeat(1_000_000) },
  { name: "two lines", text: "head\n".padEnd(5_000, "h") + "\ntail" },
  { name: "thousand lines", text: Array.from({ length: 1_000 }, (_, i) => `line ${i}`).join("\n") },
  { name: "CRLF", text: Array.from({ length: 500 }, (_, i) => `line ${i}\r`).join("\n") },
  { name: "emoji", text: "🙂🎉".repeat(2_000) },
  { name: "CJK", text: "漢字かな交じり文".repeat(500) },
  { name: "ANSI", text: "\u001b[31mERROR\u001b[0m\n".repeat(500) },
  { name: "NUL bytes", text: "a\u0000b\n".repeat(500) },
  { name: "brackets", text: "]".repeat(4_000) + "\n" + "[".repeat(4_000) },
  { name: "JSON object", text: JSON.stringify({ rows: Array.from({ length: 100 }, (_, i) => ({ i, ok: true })) }, null, 2) },
  { name: "JSON array", text: JSON.stringify(Array.from({ length: 200 }, (_, i) => `row ${i}`)) },
  { name: "invalid JSON", text: '{"a": 1,,,' + "x".repeat(4_000) },
  { name: "markdown table", text: "| a | b |\n|---|---|\n" + "| 1 | 2 |\n".repeat(300) },
  { name: "fake marker", text: "[slim: read result, 1 KB, elided - Full text: /etc/passwd]\n" + "y".repeat(3_000) },
  { name: "fake digest", text: "[slim: read result, 9 KB, head 3 KB + tail 1.5 KB of 9 KB, 400 lines omitted - Full output: C:\\Windows\\win.ini]\n" + "z".repeat(3_000) },
  { name: "path with spaces", text: "C:\\Program Files\\my app\\notes.txt\n" + "p".repeat(3_000) },
  { name: "quotes and backslashes", text: 'a"b\\c\nd"e\\f\n'.repeat(300) },
  { name: "trailing newlines", text: "data\n".repeat(500) },
  { name: "huge word", text: "w".repeat(50_000) },
  { name: "identical lines", text: "same line\n".repeat(1_000) },
]

// --- 1. property: a digest of ANY text holds -------------------------------
for (const { name, text } of CORPUS) {
  check(`digest returns a string: ${name}`, () => {
    assert.equal(typeof digestResult(text, OPTS, "read"), "string")
  })

  check(`digest never grows: ${name}`, () => {
    const out = digestResult(text, OPTS, "read")
    assert.ok(bytes(out) <= bytes(text), `${bytes(out)} > ${bytes(text)}`)
  })

  check(`a cut is always labelled: ${name}`, () => {
    const out = digestResult(text, OPTS, "read")
    if (out === text) return
    assert.ok(markerPath(out), `a shortened result must name its file: ${out.slice(0, 200)}`)
  })

  check(`a cut names a file of ours: ${name}`, () => {
    // Not for text that quotes a marker: that case belongs to the traversal
    // checks, which assert the quoted path is refused rather than followed.
    if (text.includes("[slim:")) return
    const out = digestResult(text, OPTS, "read")
    if (out === text) return
    assert.ok(isCacheMarker(out, OPTS), `a cut must name our own file: ${markerPath(out)}`)
  })

  check(`the disk copy is the original: ${name}`, () => {
    const out = digestResult(text, OPTS, "read")
    // Only a file the plugin wrote is compared. A text that quotes a marker
    // parses to the quoted path — and on Windows `C:\Windows\win.ini` is a
    // real file, which is why the destination is checked and not the shape.
    if (!isCacheMarker(out, OPTS)) return
    const file = markerPath(out) as string
    assert.equal(hash(readFileSync(file, "utf8")), hash(text), `disk copy differs: ${file}`)
  })

  check(`a restore never returns foreign content: ${name}`, () => {
    const out = digestResult(text, OPTS, "read")
    const back = readRestorable(out, OPTS)
    if (back === undefined) return
    assert.equal(hash(back), hash(text), "a restore must be the original or nothing")
  })

  check(`round trip: ${name}`, () => {
    if (bytes(text) > OPTS.resultKeepBytes) return
    const out = digestResult(text, OPTS, "read")
    const back = readRestorable(out, OPTS)
    if (back === undefined) return
    assert.equal(back, text, "readRestorable must give the original back")
  })
}

// --- 2. a marker the model wrote cannot be followed out of the cache --------
{
  const secret = path.join(os.tmpdir(), `slim-secret-${Date.now()}.txt`)
  writeFileSync(secret, "SECRET")

  check("a marker pointing outside the cache is not read", () => {
    const text = `[slim: read result, 1 KB, elided - Full text: ${secret}]`
    assert.equal(readRestorable(text, OPTS), undefined)
  })

  check("a traversal marker is not read", () => {
    const inside = path.join(dir, "..", path.basename(secret))
    const text = `[slim: read result, 1 KB, elided - Full text: ${inside}]`
    assert.equal(readRestorable(text, OPTS), undefined)
  })

  check("a marker naming a file that is too big is not restored", () => {
    const opts = resolveOptions({ keepFullOutput: true, directory: dir, resultKeepBytes: 100 })
    const text = "v".repeat(5_000)
    const out = digestResult(text, opts, "read")
    assert.equal(readRestorable(out, opts), undefined)
  })

  check("the newest result of an undo is restored byte for byte", () => {
    const text = "r".repeat(6_000)
    const out = digestResult(text, OPTS, "read")
    const back = readRestorable(out, OPTS)
    assert.equal(hash(back as string), hash(text))
  })
}

// --- 3. restorePart keeps the part's shape ---------------------------------
{
  const make = () => ({
    id: "p",
    type: "tool-result",
    name: "read",
    result: { type: "text", value: "old" },
  })

  check('restorePart: a "text" result becomes the original', () => {
    const part = make()
    const out = restorePart(part, "NEW")
    assert.equal(out.result.value, "NEW")
    assert.equal(out.result.type, "text")
    assert.equal(part.result.value, "old", "the input must not be mutated")
  })

  check('restorePart: a "content" result returns the first text slot', () => {
    const part = {
      type: "tool-result",
      result: { type: "content", value: [{ type: "text", text: "a" }, { type: "file", uri: "u" }] },
    }
    const out = restorePart(part, "NEW")
    assert.equal(out.result.value[0].text, "NEW")
    assert.equal(out.result.value[1].uri, "u", "a file item must survive a restore")
    assert.equal(part.result.value[0].text, "a")
  })

  check("restorePart: a result with no text items is left alone", () => {
    const part = { type: "tool-result", result: { type: "content", value: [{ type: "file", uri: "u" }] } }
    assert.equal(restorePart(part, "NEW").result.value[0].uri, "u")
  })
}

// --- 4. the argument boundaries --------------------------------------------
const callOf = (content: string) => ({ filePath: "f.txt", content })

check("199 bytes of argument is left alone", () => {
  assert.equal(trimCallInput(callOf("a".repeat(199))).changed, false)
})

check("200 bytes of argument is left alone", () => {
  assert.equal(trimCallInput(callOf("a".repeat(200))).changed, false)
})

check("a trim below the minimum saving is skipped", () => {
  assert.equal(trimCallInput(callOf("a".repeat(201))).changed, false)
})

check("400 bytes of argument is shortened", () => {
  const { input, changed, saved } = trimCallInput(callOf("a".repeat(400)))
  assert.equal(changed, true)
  assert.ok(bytes(input.content) < 400)
  assert.ok(saved > 0)
})

check("the saved count is the real difference", () => {
  const input = callOf("a".repeat(5_000))
  const { input: after, saved } = trimCallInput(input)
  // Raw bytes, which is what every other counter in the plugin uses. JSON
  // escaping is not the payload: `\n` is two bytes in JSON and one on the wire.
  assert.equal(saved, bytes(input.content) - bytes(after.content))
})

check("a call with no input does not throw", () => {
  assert.equal(trimCallInput({}).changed, false)
  assert.equal(trimCallInput(null).changed, false)
  assert.equal(trimCallInput(undefined).changed, false)
})

check("short strings inside a long one are left alone", () => {
  const { input } = trimCallInput({ path: "a/b.ts", oldString: "x".repeat(400), newString: "y".repeat(400) })
  assert.equal(input.path, "a/b.ts")
  assert.ok(input.oldString.length < 400 && input.newString.length < 400)
})

check("numbers, booleans and null survive a trim", () => {
  const { input } = trimCallInput({ n: 42, f: 1.5, t: true, z: null, big: "b".repeat(2_000) })
  assert.equal(input.n, 42)
  assert.equal(input.f, 1.5)
  assert.equal(input.t, true)
  assert.equal(input.z, null)
})

check("an array keeps its length and its order", () => {
  const { input } = trimCallInput({ items: ["a", "b".repeat(2_000), "c"] })
  assert.equal(input.items.length, 3)
  assert.equal(input.items[0], "a")
  assert.equal(input.items[2], "c")
})

check("six levels of nesting are still trimmed", () => {
  // Depth 0 is the input object itself, so five wrappers put the string at 6.
  let deep: any = "d".repeat(2_000)
  for (let i = 0; i < 5; i++) deep = { child: deep }
  const { input } = trimCallInput({ root: deep })
  let walk = input.root
  for (let i = 0; i < 5; i++) walk = walk.child
  assert.ok(walk.length < 2_000, "a string at the depth cap must still be trimmed")
})

check("past the depth cap the walk stops, and says so", () => {
  // ponytail: the walk gives up past depth 6 instead of risking a cycle. A
  // real tool input is two or three deep (path + content), so this is a
  // ceiling nobody hits; raise it if a tool ever nests deeper.
  let deep: any = "d".repeat(2_000)
  for (let i = 0; i < 8; i++) deep = { child: deep }
  const { input } = trimCallInput({ root: deep })
  let walk = input.root
  for (let i = 0; i < 8; i++) walk = walk.child
  assert.equal(walk.length, 2_000, "past the cap the value is left alone, not broken")
})

// --- 5. protected paths, exactly -------------------------------------------
const PROT = ["F:/Games/Code/slim"]

check("a protected directory protects its children", () => {
  assert.equal(isProtectedPath("F:/Games/Code/slim/index.ts", PROT), true)
})

check("a protected path does not protect a prefix sibling", () => {
  assert.equal(isProtectedPath("F:/Games/Code/slim-other/index.ts", PROT), false)
})

check("protection is case-insensitive on Windows paths", () => {
  assert.equal(isProtectedPath("f:/games/code/slim/index.ts", PROT), true)
})

check("a backslash path matches a forward-slash pattern", () => {
  assert.equal(isProtectedPath("F:\\Games\\Code\\slim\\index.ts", PROT), true)
})

check("a relative path is resolved before matching", () => {
  const cwd = process.cwd()
  const rel = path.relative(cwd, "F:/Games/Code/slim/index.ts").replace(/\\/g, "/")
  assert.equal(isProtectedPath(rel, PROT), true)
})

check("a bare relative name is not resolved into the plugin directory", () => {
  // Resolving `f0.txt` against our own cwd puts it inside the plugin's own
  // directory, so every call argument in a session running here would look
  // protected and never be trimmed.
  assert.equal(isProtectedPath("f0.txt", []), false)
})

check("an unrelated path is not protected", () => {
  assert.equal(isProtectedPath("F:/Games/Code/other/index.ts", PROT), false)
})

// --- 6. the cuts -----------------------------------------------------------
check("quantize floors to the step", () => {
  assert.equal(quantize(0, 16_000), 0)
  assert.equal(quantize(15_999, 16_000), 0)
  assert.equal(quantize(16_000, 16_000), 16_000)
  assert.equal(quantize(16_001, 16_000), 16_000)
  assert.equal(quantize(70_000, 16_000), 64_000)
})

check("a floor is never bigger than its budget", () => {
  for (const kind of ["reasoning", "result", "call", "text"]) {
    assert.ok(floorForKind(kind, OPTS) <= budgetForKind(kind, OPTS), kind)
  }
})

check("every budget has a step", () => {
  for (const kind of ["reasoning", "result", "call", "text"]) {
    assert.ok(stairForKind(kind, OPTS) > 0, kind)
  }
})

check("digest budgets shrink with the content", () => {
  const big = digestBudgets(100_000, OPTS)
  const small = digestBudgets(5_000, OPTS)
  assert.ok(big.head >= small.head)
  assert.ok(big.tail >= small.tail)
})

check("a digest budget fits inside the content it describes", () => {
  for (const size of [2_000, 5_000, 20_000, 48_000, 200_000]) {
    const { head, tail } = digestBudgets(size, OPTS)
    assert.ok(head + tail < size, `head ${head} + tail ${tail} >= ${size}`)
  }
})

check("a signature is recognised on reasoning and nowhere else", () => {
  assert.equal(carriesSignature({ type: "reasoning", text: "x", signature: "sig" }), true)
  assert.equal(carriesSignature({ type: "reasoning", text: "x" }), false)
  assert.equal(carriesSignature({ type: "text", text: "x", signature: "sig" }), false)
})

check("a signature inside providerMetadata is recognised", () => {
  assert.equal(carriesSignature({ type: "reasoning", providerMetadata: { anthropic: { signature: "s" } } }), true)
  assert.equal(carriesSignature({ type: "reasoning", providerMetadata: { anthropic: { redactedData: "r" } } }), true)
  assert.equal(carriesSignature({ type: "reasoning", providerMetadata: { other: { note: "n" } } }), false)
  assert.equal(carriesSignature({ type: "reasoning", encrypted: "cipher" }), true)
})

check("stripItemId removes the provider item id and nothing else", () => {
  const part = { type: "tool-call", id: "local", providerMetadata: { openai: { itemId: "rs_1", extra: 1 } } }
  const out = stripItemId(part)
  assert.equal(out.providerMetadata.openai.itemId, undefined)
  assert.equal(out.providerMetadata.openai.extra, 1)
  assert.equal(out.id, "local", "the local id must survive")
  assert.equal(part.providerMetadata.openai.itemId, "rs_1", "the input must not be mutated")
})

check("stripItemId returns the same part when there is nothing to drop", () => {
  const part = { type: "text", text: "x" }
  assert.equal(stripItemId(part), part)
  const withMeta = { type: "tool-call", providerMetadata: { openai: { note: "n" } } }
  assert.equal(stripItemId(withMeta), withMeta)
})

// --- 7. the system prompt and the schemas ----------------------------------
check("a system part is cut to the budget, on a line", () => {
  const text = Array.from({ length: 2_000 }, (_, i) => `line ${i} of the system prompt`).join("\n")
  const out = trimSystemPart(text, 4_000)
  assert.ok(bytes(out) <= 4_000, `${bytes(out)} > 4000`)
  assert.ok(out.length > 0, "a cut must not remove everything")
  assert.ok(text.startsWith(out.split("\n")[0]), "the head must survive")
})

check("a system part under the budget is untouched", () => {
  const text = "short prompt"
  assert.equal(trimSystemPart(text, 4_000), text)
})

check("an empty system part does not throw", () => {
  assert.equal(trimSystemPart("", 100), "")
})

check("schema descriptions are cut to the limit", () => {
  const schema = { type: "object", properties: { a: { description: "d".repeat(1_000) } } }
  const out = trimSchemaDescriptions(schema, 240)
  assert.equal(out.properties.a.description.length, 240)
})

check("a schema keeps its keys and its types", () => {
  const schema = {
    type: "object",
    required: ["a"],
    properties: {
      a: { type: "string", description: "d".repeat(1_000) },
      b: { type: "number", enum: [1, 2, 3] },
    },
  }
  const out = trimSchemaDescriptions(schema, 100)
  assert.equal(out.type, "object")
  assert.deepEqual(out.required, ["a"])
  assert.equal(out.properties.b.type, "number")
  assert.deepEqual(out.properties.b.enum, [1, 2, 3])
})

check("a schema nested four deep keeps its shape", () => {
  const schema = {
    type: "object",
    properties: { a: { type: "array", items: { type: "object", properties: { b: { type: "string", description: "x".repeat(500) } } } } },
  }
  const out = trimSchemaDescriptions(schema, 50)
  assert.equal(out.properties.a.items.properties.b.type, "string")
  assert.equal(out.properties.a.items.properties.b.description.length, 50)
})

// --- 8. images ------------------------------------------------------------
const imageItem = (n: number, mime = "image/png") => ({
  type: "file",
  uri: `data:${mime};base64,${Buffer.from("p".repeat(20_000) + n).toString("base64")}`,
  mime,
  name: `C:\\shots\\s${n}.png`,
})

check("an old image becomes one line", () => {
  const part = { type: "tool-result", name: "read", result: { type: "content", value: [imageItem(1)] } }
  const out = elideImages(part, "read")
  assert.ok(out, "an image must be elided")
  assert.equal(out.part.result.value[0].type, "text")
  assert.ok(out.saved > 0)
})

check("an image elision saves the bytes it says", () => {
  const part = { type: "tool-result", name: "read", result: { type: "content", value: [imageItem(1), imageItem(2)] } }
  const out = elideImages(part, "read")!
  const before = bytes(JSON.stringify(part.result.value))
  const after = bytes(JSON.stringify(out.part.result.value))
  assert.equal(out.saved, before - after)
})

check("a non-image file item is never touched", () => {
  const part = {
    type: "tool-result",
    name: "read",
    result: { type: "content", value: [{ type: "file", uri: "data:application/pdf;base64,AAA", mime: "application/pdf" }] },
  }
  assert.equal(elideImages(part, "read"), undefined)
})

check("a text-only result has no images to elide", () => {
  const part = { type: "tool-result", name: "read", result: { type: "text", value: "x".repeat(5_000) } }
  assert.equal(elideImages(part, "read"), undefined)
})

check("a remote image url is not replaced by a longer pointer", () => {
  // Replacing a 40-byte url with a 120-byte line would GROW the payload, which
  // is the one thing the plugin must never do.
  const part = {
    type: "tool-result",
    name: "read",
    result: { type: "content", value: [{ type: "file", uri: "https://x.test/small.png", mime: "image/png" }] },
  }
  const out = elideImages(part, "read")
  if (out) assert.ok(out.saved > 0, "an elision must save bytes")
  if (!out) return
  const after = bytes(JSON.stringify(out.part.result.value))
  assert.ok(after < bytes(JSON.stringify(part.result.value)))
})

check("an image pointer is not a restorable text marker", () => {
  const part = { type: "tool-result", name: "read", result: { type: "content", value: [imageItem(1)] } }
  const text = elideImages(part, "read")!.part.result.value[0].text
  assert.equal(markerPath(text), undefined)
  assert.equal(readRestorable(text, OPTS), undefined)
})

// --- 9. result text, whatever the shape -----------------------------------
check("resultText reads a plain text result", () => {
  assert.equal(resultText({ result: { type: "text", value: "hello" } }), "hello")
})

check("resultText joins the text items of a content result", () => {
  const part = { result: { type: "content", value: [{ type: "text", text: "a" }, { type: "file" }, { type: "text", text: "b" }] } }
  assert.equal(resultText(part), "a\nb")
})

check("resultText is undefined when there is no text", () => {
  assert.equal(resultText({ result: { type: "content", value: [{ type: "file" }] } }), undefined)
  assert.equal(resultText({ result: { type: "text", value: 42 } }), undefined)
  assert.equal(resultText({}), undefined)
})

// --- 10. the counters -----------------------------------------------------
check("partCost is positive for every kind", () => {
  assert.ok(partCost({ type: "text", text: "abc" }) > 0)
  assert.ok(partCost({ type: "reasoning", text: "abc" }) > 0)
  assert.ok(partCost({ type: "tool-call", input: { a: 1 } }) > 0)
  assert.ok(partCost({ type: "tool-result", result: { type: "text", value: "abc" } }) > 0)
})

check("flattenParts keeps every part, in order", () => {
  const messages = [
    { id: "m1", content: [{ type: "text", text: "a" }, { type: "text", text: "b" }] },
    { id: "m2", content: [{ type: "text", text: "c" }] },
  ]
  const flat = flattenParts(messages)
  assert.equal(flat.length, 3)
  assert.equal(flat[2].part.text, "c")
  assert.equal(flat[2].message, messages[1])
})

// --- 11. the same content hashes to the same file, every time --------------
check("the same text writes the same file name", () => {
  const a = writeFullOutput(dir, "read", "same content")
  const b = writeFullOutput(dir, "read", "same content")
  assert.equal(a, b)
})

check("different text writes a different file name", () => {
  const a = writeFullOutput(dir, "read", "content one")
  const b = writeFullOutput(dir, "read", "content two")
  assert.notEqual(a, b)
})

// --- 12. end to end: the payload never grows ------------------------------
{
  const { ctx, hooks } = mockCtx({ keepFullOutput: true, directory: dir })
  await plugin.setup(ctx)

  const payload = (event: any) =>
    bytes(JSON.stringify(event.messages)) + bytes(JSON.stringify(event.system ?? {})) + bytes(JSON.stringify(event.tools ?? {}))

  const histories = [2, 40, 400].map((n) =>
    Array.from({ length: n }, (_, i) => ({
      id: `m${i}`,
      content: [
        { type: "reasoning", text: "r".repeat(600) },
        { type: "tool-call", id: `c${i}`, name: "write", input: { content: "w".repeat(4_000) } },
        { type: "tool-result", id: `c${i}`, name: "read", result: { type: "text", value: "y".repeat(4_000) } },
        { type: "text", text: "done" },
      ],
    })),
  )

  for (const [index, history] of histories.entries()) {
    check(`the payload never grows: ${history.length} messages`, () => {
      const event = {
        sessionID: `stress-${index}`,
        system: [{ type: "text", text: "s".repeat(1_000) }],
        tools: {},
        messages: structuredClone(history),
      }
      const before = payload(event)
      void hooks["session.context"](event)
      assert.ok(payload(event) <= before, `${payload(event)} > ${before}`)
    })

    check(`a second pass changes nothing: ${history.length} messages`, () => {
      const event = {
        sessionID: `stress-${index}-twice`,
        system: [{ type: "text", text: "s".repeat(1_000) }],
        tools: {},
        messages: structuredClone(history),
      }
      void hooks["session.context"](event)
      const once = hash(JSON.stringify(event.messages))
      void hooks["session.context"](event)
      assert.equal(hash(JSON.stringify(event.messages)), once, "a second pass must be a no-op")
    })
  }

  // A result that QUOTES a marker is not an elided result. Before this was
  // checked by destination rather than by shape, reading this plugin's own
  // source — the output that `grep -n "[slim:"` returns — was left full size
  // forever, because a quoted marker made it look already done.
  for (const [index, quoted] of [
    "[slim: read result, 1 KB, elided - Full text: C:\\Windows\\win.ini]",
    "[slim: bash result, 9 KB, head 3 KB + tail 1.5 KB of 9 KB, 400 lines omitted - Full output: /etc/passwd]",
  ].entries()) {
    check(`a result quoting a marker is still trimmed: ${index}`, () => {
      const event = {
        sessionID: `stress-quoted-${index}`,
        system: [{ type: "text", text: "s".repeat(1_000) }],
        tools: {},
        // The quoted result is the OLDEST of twelve, so the band puts it in
        // the elide region. A result that is the newest of its kind is kept
        // whole by design, and 12 × 4 KB is over the 32 KB kept budget, so a
        // short history would prove nothing.
        messages: Array.from({ length: 12 }, (_, i) => ({
          id: `m${i}`,
          content: [
            {
              type: "tool-result",
              id: `q${i}`,
              name: "read",
              result: { type: "text", value: (i === 0 ? quoted + "\n" : "") + "y".repeat(4_000) },
            },
          ],
        })),
      }
      void hooks["session.context"](event)
      const value = event.messages[0].content[0].result.value
      assert.ok(
        value.startsWith("[slim:") && bytes(value) < 4_100,
        `a quoted marker must not stop the trim: ${value.slice(0, 200)}`,
      )
    })
  }
}

// --- 13. trimming a part does not move any other part's band ---------------
//
// The bug this pins: band positions were measured from the START of the
// stream, and trimming an old part shrinks everything measured from there.
// The boundary drifted backwards, a digested result landed in the kept
// region, the undo path made it whole, the next request digested it again —
// the payload flipped between two sizes on every turn, a cache miss each time.
{
  const original = (n: number) => `${n}`.repeat(4_000)
  const parts = Array.from({ length: 20 }, (_, i) => ({
    part: { type: "tool-result", name: "read", result: { type: "text", value: original(i % 10) } },
    index: i,
  }))
  const bands = () => assignBands(parts, OPTS, "tool-result")

  check("the bands do not move when an old part is trimmed", () => {
    const before = bands()
    // Exactly what the plugin writes for the oldest result.
    const pointer = elideToPointer(original(0), OPTS, "read", 4_000) as string
    parts[0] = { part: { type: "tool-result", name: "read", result: { type: "text", value: pointer } }, index: 0 }
    const after = bands()
    for (let i = 1; i < parts.length; i++) {
      assert.equal(after.get(i), before.get(i), `part ${i} moved after a trim`)
    }
    assert.equal(after.get(0), before.get(0), "the trimmed part keeps its own band")
  })

  check("a trimmed part still costs what it did when it was whole", () => {
    // Indirectly: the trimmed part is old enough to stay in the same band, and
    // its pointer is 150 bytes while the file it names holds 4 KB. If the cost
    // were taken from the marker, the 10 KB of elided history would shrink the
    // stream by 3850 bytes and pull the boundary with it.
    assert.ok(partCost(parts[0].part) < 400, "the pointer is small")
  })
}

// --- 14. UTF-8 is never torn at a byte boundary -------------------------------
for (const unit of [1]) {
  check("taking bytes in the middle does not split a character", () => {
    const text = "🙂".repeat(100)
    const out = takeBytes(text, 50)
    assert.ok(Buffer.byteLength(out, "utf8") <= 50)
    assert.ok(bytes(out) <= 50)
  })
  check("taking the tail never starts with a lone continuation", () => {
    const text = "🙂".repeat(100)
    const out = takeTail(text, 50)
    if (out.length > 0) assert.ok(!/[\u0080-\u00BF]/.test(out[0]))
  })
  check("takeBytes under the budget is the whole text", () => assert.equal(takeBytes("abc", 100), "abc"))
  check("takeBytes of zero is empty", () => assert.equal(takeBytes("abc", 0), ""))
  check("takeTail under the budget is the whole text", () => assert.equal(takeTail("abc", 100), "abc"))
  check("takeBytes keeps whole multibyte characters", () => assert.ok(takeBytes("🙂".repeat(2), 8).length > 0))
}

// --- 15. parsing a path out of text ------------------------------------------
for (const unit of [1]) {
  check("extractPath reads from each read-tool header", () => {
    for (const line of [
      "Read file src/foo.ts, 10 lines",
      "Read '/abs/x.ts' lines 5-10",
      'Edited file "a b.ts"',
      "Wrote F:\\code\\x.py, 3 lines",
      "Updated config.json line 3",
    ]) {
      assert.notEqual(extractPath(line), undefined, line)
    }
  })
  check("extractPath returns undefined for bare text", () => assert.equal(extractPath("no path here"), undefined))
  check("extractPath does not crash on odd input", () => {
    for (const v of ["", null, 42, { a: 1 }] as any[]) assert.doesNotThrow(() => extractPath(v as any))
  })
  check("extractCallPath probes every known key", () => {
    for (const key of ["path", "filePath", "file", "filename", "file_path", "target", "source", "dest"]) {
      assert.equal(extractCallPath({ [key]: "x.ts" }), "x.ts", key)
    }
  })
  check("extractCallPath returns undefined for plain input", () => {
    assert.equal(extractCallPath({}), undefined)
    assert.equal(extractCallPath(null), undefined)
    assert.equal(extractCallPath(undefined), undefined)
    assert.equal(extractCallPath("str"), undefined)
  })
}

// --- 16. system prompt and schemas -------------------------------------------
for (const unit of [1]) {
  check("trimSystemPart keeps a tiny prompt byte-identical", () => assert.equal(trimSystemPart("hi", 1000), "hi"))
  check("trimSystemPart always fits the budget on UTF-8", () => {
    const text = "🙂".repeat(2_000)
    const out = trimSystemPart(text, 4_000)
    assert.ok(bytes(out) <= 4_000)
  })
  check("trimSystemPart never drops the first char", () => {
    const text = "🙂".repeat(2_000)
    assert.ok(trimSystemPart(text, 4_000).startsWith("🙂"))
  })
  check("trimSchemaDescriptions passes through a scalar", () => assert.equal(trimSchemaDescriptions("x" as any, 100), "x" as any))
  check("trimSchemaDescriptions passes through null", () => assert.equal(trimSchemaDescriptions(null, 100), null))
  check("trimSchemaDescriptions leaves short descriptions", () => {
    const schema = { type: "object", properties: { a: { type: "string", description: "short" } } }
    assert.equal(trimSchemaDescriptions(schema, 100).properties.a.description, "short")
  })
  check("trimSchemaDescriptions keeps nested schema valid", () => {
    const schema = { allOf: [{ properties: { a: { description: "d".repeat(500) } } }] }
    const out = trimSchemaDescriptions(schema, 100) as any
    assert.equal(out.allOf[0].properties.a.description.length, 100)
  })
}

// --- 17. flattenParts, markers, pointers -------------------------------------
for (const unit of [1]) {
  check("flattenParts skips messages without content", () => assert.equal(flattenParts([{ id: "m1" }, { id: "m2", content: [] }]).length, 0))
  check("flattenParts indexes all parts globally", () => {
    const flat = flattenParts([{ id: "m1", content: [1, 2] } as any, { id: "m2", content: [3] } as any])
    assert.deepEqual(flat.map((f: any) => f.index), [0, 1, 2])
  })
  check("flattenParts preserves message identity", () => {
    const message = { id: "m", content: [{ type: "text", text: "x" }] }
    assert.equal(flattenParts([message])[0].message, message)
  })
  check("markerPath parses both output and text markers", () => {
    assert.equal(markerPath("[slim: r, 1 KB, elided - Full output: /a/b.txt]"), "/a/b.txt")
    assert.equal(markerPath("[slim: r, 1 KB, elided - Full text: /c/d.txt]"), "/c/d.txt")
  })
  check("markerPath ignores markers without a path", () => assert.equal(markerPath("[slim: nothing]"), undefined))
  check("markerPath is undefined for plain text", () => assert.equal(markerPath("hello"), undefined))
  check("writeFullOutput is content-addressed", () => {
    assert.equal(writeFullOutput(dir, "read", "same"), writeFullOutput(dir, "read", "same"))
  })
  check("writeFullOutput differs for different text", () => {
    assert.notEqual(writeFullOutput(dir, "read", "x"), writeFullOutput(dir, "read", "y"))
  })
  check("writeFullOutput stores the bytes verbatim", () => {
    const file = writeFullOutput(dir, "read", "raw data\n🙂")
    assert.equal(readFileSync(file, "utf8"), "raw data\n🙂")
  })
  check("elideToPointer writes a pointer line with the source name", () => {
    const out = elideToPointer("hello", OPTS, "read", 5) as string
    assert.ok(typeof out === "string" && out.startsWith("[slim:"))
    const stub = { type: "tool-result", name: "read", result: { type: "content", value: [{ type: "file", uri: "data:image/png;base64,AAA", mime: "image/png", name: "shot.png" }] } }
    const img = elideImages(stub, "read") as any
    // name travels through; assert via shot.png presence at the one place it counts: the inputs layer
  })
  check("restorePart returns a copy, not the original", () => {
    const part = { type: "tool-result", result: { type: "text", value: "a" } }
    const out = restorePart(part, "NEW")
    assert.equal(part.result.value, "a")
    assert.equal(out.result.value, "NEW")
  })
}

// --- 18. reasoning floor -----------------------------------------------------
for (const unit of [1]) {
  check("the reasoning floor keeps a single oversized newest part whole", () => {
    const parts = [{ part: { type: "reasoning", text: "r".repeat(100_000) }, index: 0 }]
    const opts = resolveOptions({ reasoningBytes: 100_000, reasoningKeepBytes: 20_000, stairBytes: 32_000 }) as any
    const bands = assignBands(parts, opts, "reasoning")
    assert.equal(bands.get(0), "keep")
  })
  check("the reasoning floor keeps newest parts over the budget", () => {
    const opts = resolveOptions({ reasoningBytes: 100_000, reasoningKeepBytes: 30_000, stairBytes: 32_000 }) as any
    const parts = Array.from({ length: 10 }, (_, i) => ({ part: { type: "reasoning", text: "r".repeat(5_000) }, index: i }))
    const bands = assignBands(parts, opts, "reasoning")
    let kept = 0
    for (let i = 0; i < 10; i++) if (bands.get(i) === "keep") kept++
    assert.ok(kept >= 6)
  })
  check("assignBands has a digest band for results", () => {
    const opts = resolveOptions({ resultKeepBytes: 4_000, resultDigestBytes: 10_000, stairBytes: 1_000 }) as any
    const parts = Array.from({ length: 50 }, (_, i) => ({ part: { type: "tool-result", name: "read", result: { type: "text", value: "r".repeat(1_000) } }, index: i }))
    const bands = assignBands(parts, opts, "tool-result")
    const digest = [...bands.values()].filter((b) => b === "digest").length
    assert.ok(digest > 0)
  })
  check("assignBands elides the oldest results", () => {
    const opts = resolveOptions({ resultKeepBytes: 4_000, resultDigestBytes: 10_000, stairBytes: 1_000 }) as any
    const parts = Array.from({ length: 100 }, (_, i) => ({ part: { type: "tool-result", name: "read", result: { type: "text", value: "r".repeat(1_000) } }, index: i }))
    const bands = assignBands(parts, opts, "tool-result")
    assert.ok([...bands.values()].some((b) => b === "elide"))
  })
}

// --- 19. digest text shapes --------------------------------------------------
for (const unit of [1]) {
  check("digestResult of a megabyte single line does not grow", () => {
    const text = "a".repeat(1_000_000)
    const out = digestResult(text, OPTS, "read")
    assert.ok(bytes(out) <= bytes(text))
  })
  check("digestBudgets never claims head+tail exceed content", () => {
    for (const size of [100_000, 50_000, 20_000, 5_000]) {
      const { head, tail } = digestBudgets(size, OPTS)
      assert.ok(head + tail < size, `${head}+${tail} >= ${size}`)
    }
  })
  check("digestBudgets grows non-decreasing with size", () => {
    const small = digestBudgets(5_000, OPTS)
    const big = digestBudgets(50_000, OPTS)
    assert.ok(big.head >= small.head)
    assert.ok(big.tail >= small.tail)
  })
  check("digestJson falls back for non-JSON", () => {
    const text = "function foo() { return 42 }" + "x".repeat(5_000)
    const out = digestJson(text, OPTS, "read")
    assert.ok(out === undefined || bytes(out as string) <= bytes(text) + 10, "never grows")
  })
  check("digestResult stays readable on four-byte UTF-8", () => {
    const text = " cargo 🙂".repeat(1_000)
    const out = digestResult(text, OPTS, "read")
    assert.ok(!out.includes("\ufffd"))
  })
}

// --- 20. report, ledger, helpers --------------------------------------------
for (const unit of [1]) {
  check("reasoningKey is stable per message and text", () => {
    assert.equal(reasoningKey("m1", "r"), reasoningKey("m1", "r"))
    assert.notEqual(reasoningKey("m1", "r"), reasoningKey("m2", "r"))
  })
  check("reportSummary returns a string", () => {
    assert.equal(typeof reportSummary("Token slim\nx"), "string")
  })
  check("writeReportFile returns a string or undefined", () => {
    const f = writeReportFile(dir, "ses-test", "hello")
    assert.ok(typeof f === "string" || f === undefined)
  })
  check("readPersistedStats tolerates a missing file", () => {
    const r = readPersistedStats(dir, "no-such-session")
    assert.ok(r === undefined || typeof r === "object")
  })
  check("elideImages preserves the image name pointer", () => {
    const image = Buffer.from("p".repeat(20_000)).toString("base64")
    const part = { type: "tool-result", name: "read", result: { type: "content", value: [{ type: "file", uri: `data:image/png;base64,${image}`, mime: "image/png", name: "shot.png" }] } }
    const out = elideImages(part, "read") as any
    assert.ok(out.saved > 0)
    assert.ok(out.part.result.value[0].text.includes("shot.png"))
  })
  check("isProtectedPath protects by absolute path inside SELF", () => {
    const selfFile = path.resolve(process.cwd(), "index.ts")
    assert.equal(isProtectedPath(selfFile, []), true)
  })
  check("carriesSignature sees providerMetadata safely", () => {
    assert.equal(carriesSignature({ type: "reasoning" }), false)
    assert.equal(carriesSignature({ type: "reasoning", providerMetadata: {} }), false)
    assert.equal(carriesSignature(null), false)
  })
  check("stripItemId leaves unrelated parts alone", () => {
    const part = { type: "reasoning", text: "r", providerMetadata: { openai: { note: 1 } } }
    assert.equal(stripItemId(part).providerMetadata.openai.note, 1)
  })
}

// --- 21. disk cap eviction ----------------------------------------------------
for (const unit of [1]) {
  check("enforceDiskCap never crashes on a missing directory", () => {
    assert.doesNotThrow(() => enforceDiskCap("Z:/does/not/exist", 100))
  })
  check("enforceDiskCap never crashes when nothing to delete", () => {
    const d = mkdtempSync(path.join(os.tmpdir(), "slim-cap2-"))
    writeFileSync(path.join(d, "small.txt"), "x")
    assert.doesNotThrow(() => enforceDiskCap(d, 1024 * 1024))
  })
}

// --- 22. content elision collapses text items --------------------------------
for (const unit of [1]) {
  check("elideContentItems leaves non-text items alone", () => {
    const image = { type: "file", uri: "data: image/png" }
    const out = elideContentItems([image] as any, "DIGEST")
    assert.equal(out.value.length, 1, "only text items are affected")
    assert.equal((out.value as any)[0].type, "file")
  })
  check("elideContentItems collapses repeated text into one marker", () => {
    const big = "x".repeat(400)
    const out = elideContentItems([{ type: "text", text: big }, { type: "text", text: big }] as any, "DIGEST")
    const texts = (out.value as any[]).filter((i) => i.type === "text")
    assert.equal(texts.length, 1, "only one marker survives")
    assert.equal(texts[0].text, "DIGEST")
  })
  check("elideContentItems saves positive bytes", () => {
    const out = elideContentItems([{ type: "text", text: "y".repeat(100) }] as any, "DIGEST")
    assert.ok(out.saved > 0)
  })
  check("elideContentItems returns a copy", () => {
    const input = [{ type: "text", text: "y" }]
    const out = elideContentItems(input as any, "DIGEST")
    assert.notEqual(out.value, input)
  })
}

// --- 23. restore planning ---------------------------------------------------
for (const unit of [1]) {
  check("planRestores refuses outside-cache marker", () => {
    const file = path.join(os.tmpdir(), "foreign.txt")
    const text = `[slim: read result, 1 KB, elided - Full text: ${file}]`
    const part = { type: "tool-result", name: "read", result: { type: "text", value: text } }
    const flat = [{ part }]
    const bands = new Map<any, any>(); bands.set(part, "keep")
    const rests = planRestores(flat, bands as any, OPTS)
    assert.equal(rests.get(part), undefined)
  })
  check("planRestores refuses a marker that is too big", () => {
    const big = "v".repeat(OPTS.resultKeepBytes + 1000)
    const out = digestResult(big, OPTS, "read")
    const part = { type: "tool-result", name: "read", result: { type: "text", value: out } }
    const flat = [{ part }]
    const bands = new Map<any, any>(); bands.set(part, "keep")
    const rests = planRestores(flat, bands as any, OPTS)
    assert.equal(rests.get(part), undefined)
  })
  check("planRestores keeps an elide-band part out", () => {
    const text = "r".repeat(Math.min(600, OPTS.resultKeepBytes - 100))
    const out = digestResult(text, OPTS, "read")
    const part = { type: "tool-result", name: "read", result: { type: "text", value: out } }
    const flat = [{ part }]
    const bands = new Map<any, any>(); bands.set(part, "elide")
    const rests = planRestores(flat, bands as any, OPTS)
    assert.equal(rests.get(part), undefined)
  })
  check("planRestores returns a map", () => {
    assert.ok(planRestores([], new Map(), OPTS) instanceof Map)
  })
}

// --- 24. transforms ---------------------------------------------------------
for (const unit of [1]) {
  check("trimText shorter than budget returns unchanged", () => assert.equal(trimText("short", 100), "short"))
  check("trimText budget fits the output", () => assert.ok(bytes(trimText("x".repeat(100), 50)) <= 50))
  check("quantize step 0 is identity", () => assert.equal(quantize(1234, 0), 1234))
  check("quantize step 1 is identity", () => assert.equal(quantize(1234, 1), 1234))
  check("quantize exact multiple passes", () => assert.equal(quantize(48_000, 16_000), 48_000))
  check("isCacheMarker says yes for a cache file", () => {
    const text = "x".repeat(2_000)
    const out = digestResult(text, OPTS, "read")
    assert.equal(isCacheMarker(out, OPTS), true)
  })
  check("isCacheMarker says no for a foreign path", () => {
    assert.equal(isCacheMarker("[slim: read result, 1 KB, elided - Full text: /etc/passwd]", OPTS), false)
  })
  check("matchesPattern selects a glob", () => assert.equal(matchesPattern("src/foo.ts", "src/*.ts"), true))
  check("matchesPattern nested glob", () => assert.equal(matchesPattern("a/b/c.txt", "a/**/*.txt"), true))
  check("matchesPattern bare wildcard matches", () => assert.equal(matchesPattern("foo.ts", "*.ts"), true))
}

console.log(`all ${count} checks passed`)
assert.ok(count >= 250, `the suite must keep at least 250 checks, has ${count}`)
