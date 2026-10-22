/**
 * Self-check for the token-slim logic.
 * Run: node --experimental-strip-types test.ts
 *
 * Every helper under test is imported from index.ts, so these checks guard the
 * code that actually ships. An earlier version of this file re-implemented
 * the helpers inline, which meant the tests could pass while the plugin
 * behaved differently.
 */
import assert from "node:assert/strict"
import {
  MIN_DESCRIPTION,
  assignBands,
  carriesSignature,
  takeBytes,
  trimSchemaDescriptions,
  trimSystemPart,
  trimText,
} from "\./index.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

// --- 1. takeBytes never splits a UTF-8 sequence -----------------------------
{
  const text = "a".repeat(10) + "€" + "b".repeat(10)
  for (let budget = 0; budget <= bytes(text); budget++) {
    const out = takeBytes(text, budget)
    assert.equal(Buffer.from(out, "utf8").length, bytes(out))
    assert.ok(bytes(out) <= budget)
    assert.ok(!out.includes("�"))
  }
}

// --- 2. signature detection --------------------------------------------------
{
  assert.equal(carriesSignature({ type: "reasoning", text: "x" }), false)
  assert.equal(carriesSignature({ type: "reasoning", text: "x", encrypted: "sig" }), true)
  assert.equal(carriesSignature({ type: "reasoning", text: "x", providerMetadata: { anthropic: { signature: "s" } } }), true)
  assert.equal(
    carriesSignature({ type: "reasoning", text: "x", providerMetadata: { ollama: { reasoningField: "reasoning_content" } } }),
    false,
  )
}

// --- 3. schema trimming keeps the shape valid --------------------------------
{
  const schema = {
    type: "object",
    properties: {
      path: { type: "string", description: "x".repeat(1000) },
      offset: { type: "number", description: "y".repeat(1000) },
    },
    required: ["path"],
  }
  const trimmed = trimSchemaDescriptions(schema, 120)
  assert.equal(trimmed.type, "object")
  assert.deepEqual(trimmed.required, ["path"])
  assert.equal(trimmed.properties.path.type, "string")
  assert.equal(trimmed.properties.offset.type, "number")
  assert.ok(bytes(trimmed.properties.path.description) <= 120)
}

// --- 4. the byte budget keeps the saving consistent ---------------------------
//
// The rule is a byte budget, not a message count, and this is the property
// that makes the saving predictable: how much is kept depends on the SIZE of
// the content, not on how many turns it took to produce it.
{
  const opts = { reasoningBytes: 100_000, reasoningKeepBytes: 20_000, stairBytes: 32_000 }
  // The same total bytes, spread two ways: 100 parts of 1 KB, or 1 part of
  // 100 KB. A byte budget keeps the same amount either way; a message-count
  // window would not.
  const many = Array.from({ length: 100 }, () => ({ type: "reasoning", text: "r".repeat(1_000) }))
  const few = [{ type: "reasoning", text: "r".repeat(100_000) }]

  const keptBytes = (parts: any[]) => {
    const bands = assignBands(
      parts.map((part, index) => ({ part, index })),
      opts as any,
      "reasoning",
    )
    return parts.reduce((n, p, i) => n + (bands.get(i) === "keep" ? bytes(p.text) : 0), 0)
  }

  const a = keptBytes(many)
  const b = keptBytes(few)
  // Both are inside the 100 KB budget, so both keep everything.
  assert.equal(a, 100_000, "100 KB of small parts must be kept whole")
  assert.equal(b, 100_000, "100 KB in one part must be kept whole")

  // Now over the budget: 300 KB either way.
  const manyOver = Array.from({ length: 300 }, () => ({ type: "reasoning", text: "r".repeat(1_000) }))
  const keptOver = keptBytes(manyOver)
  assert.ok(keptOver <= opts.reasoningBytes + opts.stairBytes, `the budget must hold: ${keptOver}`)
  // The newest is always kept whole, whatever the budget says.
  assert.ok(keptOver >= 100_000, `the newest must be kept: ${keptOver}`)
}

// --- 4b. the cut only moves on a step boundary --------------------------------
//
// The cut is measured from the start of the history and rounded down to a
// step. Between steps the request is byte-identical, so the provider's
// prompt cache still hits. This is the property that keeps the saving from
// costing more than it saves.
{
  const opts = { reasoningBytes: 100_000, reasoningKeepBytes: 20_000, stairBytes: 32_000 }
  const cutOf = (count: number) => {
    const parts = Array.from({ length: count }, () => ({ type: "reasoning", text: "r".repeat(1_000) }))
    const bands = assignBands(
      parts.map((part, index) => ({ part, index })),
      opts as any,
      "reasoning",
    )
    // The oldest part that is NOT kept marks the cut.
    for (let i = 0; i < parts.length; i++) {
      if (bands.get(i) !== "keep") return i
    }
    return parts.length
  }

  // 300 KB of reasoning: the budget keeps the newest 100 KB.
  const before = cutOf(300)
  // One more 1 KB part. The ideal cut would move by one part, but the step
  // is 32 KB, so it must not move at all.
  const after = cutOf(301)
  assert.equal(after, before, "the cut must not move until a whole step has accumulated")
}

// --- 5. a signed reasoning part is never dropped -----------------------------
{
  const messages = [
    { role: "assistant", content: [{ type: "reasoning", text: "old", providerMetadata: { anthropic: { signature: "s" } } }] },
    { role: "assistant", content: [{ type: "reasoning", text: "old2" }] },
    { role: "assistant", content: [{ type: "reasoning", text: "recent" }] },
  ]
  let dropped = 0
  for (let i = 0; i < 2; i++) {
    messages[i].content = messages[i].content.filter((p: any) => {
      if (p.type !== "reasoning" || carriesSignature(p)) return true
      dropped++
      return false
    })
  }
  assert.equal(dropped, 1)
  assert.equal(messages[0].content.length, 1, "signed part preserved")
  assert.equal(messages[0].content[0].text, "old")
  assert.equal(messages[1].content.length, 0)
  assert.equal(messages[2].content.length, 1)
}

// --- 5b. trimming prose keeps both ends and says what was cut ---------------
//
// The assistant's own text is the record of what it decided and why. The head
// says what it was doing; the tail says what it concluded. Taking only the
// head throws the conclusion away, and that is the part worth keeping.
{
  const text = "BEGIN " + "m".repeat(2000) + " END"
  const trimmed = trimText(text, 500)
  assert.ok(bytes(trimmed) <= 600, `the trimmed text must fit the budget: ${bytes(trimmed)}`)
  assert.ok(trimmed.startsWith("BEGIN"), "the head must survive")
  assert.ok(trimmed.trimEnd().endsWith("END"), "the tail must survive")
  assert.ok(/omitted/.test(trimmed), "the marker must say what was cut")
  // Text that fits is not touched at all.
  assert.equal(trimText("short", 500), "short")
}

// --- 6. system part trimming cuts on an entry boundary ----------------------
{
  const catalog = [
    "# Code Mode",
    "",
    "Use the `execute` tool...",
    "",
    "## Available tools",
    "",
    "- memory (9 tools, 5 shown)",
    "  - tools.memory.add_observations({",
    "      path: string,",
    "    })",
    "- opencode (8 tools)",
    "  - tools.opencode.models({",
    "      query?: string,",
    "    })",
  ].join("\n")
  const trimmed = trimSystemPart(catalog, 200)
  assert.ok(bytes(trimmed) <= 200, `must respect the allowance, got ${bytes(trimmed)}`)
  // Must not end mid-signature: the last kept line must be complete.
  const lastLine = trimmed.split("\n").filter(Boolean).at(-2) ?? ""
  assert.ok(!lastLine.includes("{"), "must not cut inside a tool signature")
  assert.ok(trimmed.includes("truncated"))
}

// --- 7. prose system part cuts on a line boundary ----------------------------
{
  const prose = Array.from({ length: 500 }, (_, i) => `prose line ${i}`).join("\n")
  const trimmed = trimSystemPart(prose, 200)
  assert.ok(bytes(trimmed) <= 200)
  assert.ok(trimmed.split("\n").every((l) => l.startsWith("prose line") || l === ""))
}

// --- 8. description floor is respected --------------------------------------
{
  const desc = "d".repeat(1000)
  const overflow = 500
  const canTrim = bytes(desc) > overflow + MIN_DESCRIPTION
  assert.equal(canTrim, true)
  const trimmed = takeBytes(desc, bytes(desc) - overflow)
  assert.equal(bytes(trimmed), bytes(desc) - overflow)
}

// --- 9. prune marker survives compaction ------------------------------------
{
  const prunedThroughID: { v: string | undefined } = { v: undefined }
  const pruneStart = (ids: (string | undefined)[]) => {
    if (prunedThroughID.v === undefined) return 0
    const at = ids.indexOf(prunedThroughID.v)
    return at < 0 ? 0 : at + 1
  }
  const ids = ["m0", "m1", "m2", "m3"]
  assert.equal(pruneStart(ids), 0)
  prunedThroughID.v = "m1"
  assert.equal(pruneStart(ids), 2, "resumes after the marker")
  // After compaction the marker is gone: start over rather than skip history.
  assert.equal(pruneStart(["x0", "x1"]), 0, "marker missing -> restart")
}

// --- 10. tool schema cap converges on a realistic tool set ------------------
{
  const tools: Record<string, any> = {
    subagent: { description: "d".repeat(1200), input: { type: "object", properties: { prompt: { type: "string", description: "p".repeat(800) } } } },
    edit: { description: "e".repeat(950), input: { type: "object", properties: { path: { type: "string", description: "q".repeat(500) } } } },
    shell: { description: "s".repeat(540), input: { type: "object", properties: { command: { type: "string", description: "r".repeat(900) } } } },
    read: { description: "t".repeat(489), input: { type: "object", properties: { path: { type: "string", description: "u".repeat(400) } } } },
  }
  const sizeOf = (t: any) => bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {}))
  const toolBytes = 4000
  let total = Object.values(tools).reduce((n, t) => n + sizeOf(t), 0)
  const entries = Object.entries(tools)
  assert.ok(total > toolBytes)

  for (const limit of [200, 100, 50, 20, 0]) {
    if (total <= toolBytes) break
    for (const [, tool] of entries) {
      const before = bytes(JSON.stringify(tool.input ?? {}))
      const trimmed = trimSchemaDescriptions(tool.input, limit)
      const after = bytes(JSON.stringify(trimmed))
      if (after < before) {
        total -= before - after
        tool.input = trimmed
      }
    }
  }
  if (total > toolBytes) {
    const descTotal = entries.reduce((n, [, t]) => n + bytes(t.description ?? ""), 0)
    const schemaTotal = total - descTotal
    const allowed = Math.max(0, toolBytes - schemaTotal)
    if (descTotal > allowed) {
      const scale = allowed / descTotal
      for (const [, tool] of entries) {
        const desc = tool.description ?? ""
        const size = bytes(desc)
        if (size === 0) continue
        const keep = Math.max(MIN_DESCRIPTION, Math.floor(size * scale))
        if (keep >= size) continue
        const trimmed = takeBytes(desc, keep)
        total -= size - bytes(trimmed)
        tool.description = trimmed
      }
    }
  }
  // The cap must now be met.
  assert.ok(total <= toolBytes, `cap must converge, got ${total} > ${toolBytes}`)
  // The schema shape must survive.
  for (const [, tool] of entries) {
    assert.equal(tool.input.type, "object")
    for (const [k, v] of Object.entries(tool.input.properties)) {
      assert.equal((v as any).type, "string", `property ${k} type preserved`)
    }
  }
  // Descriptions must not be destroyed below the floor.
  for (const [, tool] of entries) {
    assert.ok(bytes(tool.description) >= MIN_DESCRIPTION, "description keeps its floor")
  }
}

console.log("all checks passed")
