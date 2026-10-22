/**
 * Regression tests for the bugs found in production.
 *
 * These drive the REAL plugin — the actual `context` hook — rather than a
 * re-implementation of it. A test that re-implements the logic passes while
 * the shipped code behaves differently, which is how the earlier version of
 * this file let three bugs through.
 *
 * The bugs:
 *
 *  1. The `context` hook receives a FRESH COPY of the history on every
 *     request, and nothing it does is persisted. A prune marker that assumed
 *     persistence let the reasoning come back on the next request, so the
 *     payload never actually shrank.
 *
 *  2. `tool.execute.after` only ever sees NEW results, so oversized results
 *     already in the history stayed full size.
 *
 *  3. A digest written into every slot of a multi-item result multiplied the
 *     digest instead of replacing the content, and the payload GREW.
 */
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import plugin, { digestResult, markerPath, resolveOptions } from "./index.ts"

const bytes = (s) => Buffer.byteLength(s, "utf8")

function mockCtx(options = {}) {
  const hooks = {}
  const synthetic = []
  const commands = []
  const ctx = {
    options: { keepFullOutput: false, ...options },
    tool: { hook: async (n, f) => { hooks[`tool.${n}`] = f } },
    session: {
      hook: async (n, f) => { hooks[`session.${n}`] = f },
      synthetic: async (e) => { synthetic.push(e) },
    },
    command: { transform: async (f) => { f({ add: (c) => commands.push(c) }) } },
  }
  return { ctx, hooks, synthetic, commands }
}

/** Drive the real context hook over a history. */
async function runContext(ctx, hooks, sessionID, messages, extra = {}) {
  const event = {
    sessionID,
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
    ...extra,
  }
  await hooks["session.context"](event)
  return event
}

const reasoningCount = (messages) =>
  messages.flatMap((m) => m.content).filter((p) => p.type === "reasoning").length

// --- BUG 1: the pruned reasoning must STAY pruned ---------------------------
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  // Enough reasoning to blow past the budget.
  const persisted = Array.from({ length: 200 }, (_, i) => ({
    id: "m" + i,
    content: [{ type: "reasoning", text: "r".repeat(5_000) }],
  }))

  const e1 = await runContext(ctx, hooks, "regress1", persisted)
  const sent1 = reasoningCount(e1.messages)
  assert.ok(sent1 < persisted.length, "request 1 must prune something")

  // Request 2: a FRESH copy of the same persisted history.
  const e2 = await runContext(ctx, hooks, "regress1", persisted)
  const sent2 = reasoningCount(e2.messages)
  assert.equal(sent2, sent1, "request 2 must send the SAME amount as request 1")
  assert.ok(sent2 < persisted.length, "request 2 must still be pruned")
}

// --- BUG 1b: a growing session keeps staying pruned --------------------------
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  const persisted = Array.from({ length: 200 }, (_, i) => ({
    id: "m" + i,
    content: [{ type: "reasoning", text: "r".repeat(5_000) }],
  }))

  // The bound that matters is BYTES, not the number of parts. A byte budget
  // holds a fixed amount of content, so as tiny new parts arrive more of
  // them fit — the count can grow while the payload does not. What must not
  // grow is the payload.
  const sentBytes = []
  for (let turn = 0; turn < 10; turn++) {
    persisted.push({ id: "x" + turn, content: [{ type: "reasoning", text: "new-" + turn }] })
    const e = await runContext(ctx, hooks, "regress1b", persisted)
    sentBytes.push(
      e.messages
        .flatMap((m) => m.content)
        .filter((p) => p.type === "reasoning")
        .reduce((n, p) => n + bytes(p.text ?? ""), 0),
    )
  }
  const first = sentBytes[0]
  const last = sentBytes[sentBytes.length - 1]
  // The payload must stay within the budget plus one step of slack.
  assert.ok(last <= 80_000 + 16_000, `the sent bytes must stay bounded: ${last}`)
  assert.ok(last >= first, `the payload must not shrink as the session grows: ${last} vs ${first}`)
}

// --- BUG 2: oversized tool results in the history must be elided -------------
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  const big = "x".repeat(20_000)
  const small = "y".repeat(100)
  // The newest result is kept whole — it is the model's current context — so
  // the results that must be elided need something newer than them to push
  // them out of the keep band.
  const messages = [
    { id: "a", content: [{ type: "tool-result", name: "read", result: { type: "text", value: big } }] },
    { id: "b", content: [{ type: "tool-result", name: "read", result: { type: "text", value: small } }] },
    { id: "c", content: [{ type: "tool-result", name: "read", result: { type: "content", value: [{ type: "text", text: big }] } }] },
    { id: "d", content: [{ type: "tool-result", name: "read", result: { type: "error", value: big } }] },
    { id: "e", content: [{ type: "tool-result", name: "read", result: { type: "text", value: "z".repeat(60_000) } }] },
  ]

  const event = await runContext(ctx, hooks, "regress2", messages)
  const m = event.messages

  // The oversized text result is elided.
  assert.ok(bytes(m[0].content[0].result.value) < bytes(big), "big text result elided")
  // The small one is untouched: a digest of it would be bigger than it is.
  assert.equal(m[1].content[0].result.value, small, "small result untouched")
  // The content result is elided, and the replacement goes into the first
  // text slot.
  assert.ok(bytes(m[2].content[0].result.value[0].text) < bytes(big), "content result elided")
  // An error result is never rewritten: the model needs it verbatim.
  assert.equal(m[3].content[0].result.value, big, "an error result is never rewritten")
  // The newest result is kept whole: the model is looking at it right now.
  assert.equal(m[4].content[0].result.value.length, 60_000, "the newest result is kept whole")
}

// --- BUG 3: a multi-item result must not have its digest multiplied ----------
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  // One result split across 40 text items. The digest must replace them all,
  // not be written into every slot. A newer result pushes it out of the keep
  // band, so it is elided.
  const items = Array.from({ length: 40 }, () => ({ type: "text", text: "z".repeat(2_000) }))
  const messages = [
    {
      id: "multi",
      content: [{ type: "tool-result", name: "read", result: { type: "content", value: items } }],
    },
    {
      id: "newer",
      content: [{ type: "tool-result", name: "read", result: { type: "text", value: "n".repeat(60_000) } }],
    },
  ]

  const before = bytes(JSON.stringify(messages[0].content[0].result))
  const event = await runContext(ctx, hooks, "regress3", messages)
  const after = bytes(JSON.stringify(event.messages[0].content[0].result))
  assert.ok(after < before, `the payload must shrink: ${after} !< ${before}`)
  // The replacement is in ONE slot, not forty.
  const value = event.messages[0].content[0].result.value
  assert.equal(value.length, 1, `the digest must replace the items, not join them: ${value.length} slots`)
}

// --- BUG 2b: a file entry inside a content result must survive ---------------
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  const big = "x".repeat(20_000)
  const messages = [
    {
      id: "withfile",
      content: [
        {
          type: "tool-result",
          name: "read",
          result: {
            type: "content",
            value: [
              { type: "text", text: big },
              { type: "file", uri: "data:application/pdf;base64,AAA", mime: "application/pdf" },
              // 3 KB of base64: a 40-byte data uri would be smaller than the
              // pointer that replaces it, and the plugin must never grow.
              { type: "file", uri: "data:image/png;base64," + "A".repeat(3_000), mime: "image/png", name: "shot.png" },
            ],
          },
        },
      ],
    },
    {
      id: "newer",
      content: [{ type: "tool-result", name: "read", result: { type: "text", value: "n".repeat(60_000) } }],
    },
  ]

  const event = await runContext(ctx, hooks, "regress2b", messages)
  const value = event.messages[0].content[0].result.value
  assert.ok(bytes(value[0].text) < bytes(big), "text elided")
  assert.equal(value[1].type, "file", "a non-image file entry survives")
  assert.equal(value[1].uri, "data:application/pdf;base64,AAA", "file uri intact")
  assert.equal(value[2].type, "text", "an image entry becomes a pointer")
  assert.ok(value[2].text.includes("shot.png"), "the image pointer names its source")
}

// --- BUG 4: old tool calls must be trimmed even when each one is small ------
//
// The budget for call arguments is a budget over the WHOLE stream of calls.
// The trimmer applied the same number again as a per-call ceiling, so every
// call under 8 KB was left whole: a session with two thousand 7 KB calls
// replayed all 14 MB of them, and the plugin reported them as "kept".
{
  const { ctx, hooks } = mockCtx()
  await plugin.setup(ctx)

  const messages = Array.from({ length: 200 }, (_, i) => ({
    id: `m${i}`,
    content: [
      { type: "tool-call", name: "write", input: { path: `f${i}.txt`, content: "w".repeat(7_000) } },
      { type: "tool-result", name: "write", result: { type: "text", value: "ok" } },
    ],
  }))
  const before = messages.reduce((n, m) => n + bytes(JSON.stringify(m.content)), 0)
  const event = await runContext(ctx, hooks, "regress4", messages)
  const after = event.messages.reduce((n, m) => n + bytes(JSON.stringify(m.content)), 0)
  assert.ok(after < before / 2, `most of the call arguments must go: ${after} !< ${before / 2}`)

  // The call itself survives, with its shape.
  const old = event.messages[50].content[0]
  assert.equal(old.name, "write", "the call must survive")
  assert.equal(typeof old.input.path, "string", "the call must keep its shape")

  // The newest call is the model's current context, so it is kept whole.
  const newest = event.messages.at(-1).content[0]
  assert.equal(bytes(newest.input.content), 7_000, "the newest call is kept whole")
}

// --- BUG 5: a result undo brings back into the kept region comes back whole --
//
// The source trim persists a digest, so an undo cannot get the original back
// from the session: the marker is all that is left. The full text is on disk,
// so the marker is undone too — when the result is inside the kept region
// again, which is exactly where an undo puts it. Bounded by the region: a
// result bigger than the region stays a digest the model can read in slices.
{
  const dir = `${process.env.TEMP}\\slim-undo-${Date.now()}`
  const opts = resolveOptions({ keepFullOutput: true, directory: dir })
  const { ctx, hooks, synthetic, commands } = mockCtx({ keepFullOutput: true, directory: dir })
  await plugin.setup(ctx)

  // What the tool hook persisted for a 20 KB read: a digest, not the text.
  const big = "z".repeat(20_000)
  const digest = digestResult(big, opts, "read")
  assert.ok(digest.includes("Full output:"), "the digest must point at the file")

  // The reverted history ends with that digest. It is the newest result, so
  // it is inside the kept region and must be whole again.
  const persisted = [
    {
      id: "m1",
      content: [
        { type: "tool-result", id: "prt_1", name: "read", result: { type: "text", value: digest } },
      ],
    },
  ]
  const e1 = await runContext(ctx, hooks, "undo", persisted)
  assert.equal(
    e1.messages[0].content[0].result.value,
    big,
    "the marker must be undone from the full text on disk",
  )
  assert.equal(persisted[0].content[0].result.value, digest, "the persisted history must not be touched")

  // Same history, next request: same bytes, or the prompt cache misses.
  const e2 = await runContext(ctx, hooks, "undo", persisted)
  assert.equal(
    e2.messages[0].content[0].result.value,
    big,
    "the restore must be the same on the next request",
  )

  // The restore gives bytes back, so it is reported on its own line and the
  // percentage is not negative.
  await commands[0].execute({ sessionID: "undo" })
  const description = String(synthetic.at(-1).description)
  const file = description.match(/full report: ([^)\s]+)/)?.[1]
  assert.ok(file, `the report must have a full report file:\n${description}`)
  const full = readFileSync(file, "utf8")
  assert.ok(
    /restored:\s+-\d[\d.]* KB \(1 results/.test(full),
    `the report must show the restored bytes, once:\n${full}`,
  )
  assert.ok(!/-\d+% smaller/.test(description), `the percentage must not be negative:\n${description}`)

  // A result bigger than the whole kept region stays a digest: the restore is
  // a repair inside the budget, not a way around it.
  const huge = "h".repeat(40_000)
  const hugeDigest = digestResult(huge, opts, "read")
  const e3 = await runContext(ctx, hooks, "undo-huge", [
    {
      id: "m2",
      content: [
        { type: "tool-result", id: "prt_2", name: "read", result: { type: "text", value: hugeDigest } },
      ],
    },
  ])
  assert.equal(
    e3.messages[0].content[0].result.value,
    hugeDigest,
    "a result bigger than the kept region must stay a digest",
  )

  // A digest old enough for the digest band is not digested a second time:
  // the second marker would point at a file holding the first marker, and the
  // model would need two hops to reach the data.
  const dir2 = `${process.env.TEMP}\\slim-undo-two-${Date.now()}`
  const tiny = {
    keepFullOutput: true,
    directory: dir2,
    resultKeepBytes: 1_000,
    stairBytes: 1_000,
  }
  const { ctx: ctx2, hooks: hooks2 } = mockCtx(tiny)
  await plugin.setup(ctx2)
  const old = digestResult("q".repeat(5_000), resolveOptions(tiny), "read")
  const e4 = await runContext(ctx2, hooks2, "undo-digest", [
    {
      id: "m1",
      content: [
        { type: "tool-result", id: "prt_3", name: "read", result: { type: "text", value: old } },
      ],
    },
    {
      id: "m2",
      content: [
        {
          type: "tool-result",
          id: "prt_4",
          name: "read",
          result: { type: "text", value: "new".repeat(2_000) },
        },
      ],
    },
  ])
  assert.equal(
    e4.messages[0].content[0].result.value,
    old,
    "an existing digest must not be digested again",
  )
}

// --- BUG 6: a part is counted once, not once per request --------------------
//
// Every request re-sends the same history, so a counter bumped inside the trim
// loop is multiplied by the number of requests. Reasoning and results were
// deduped from the start; call arguments, own text and the protected skips
// were not, and the report read them next to the deduped ones as if they
// meant the same thing.
{
  const { ctx, hooks, synthetic, commands } = mockCtx({
    keepFullOutput: true,
    directory: `${process.env.TEMP}\\slim-count-${Date.now()}`,
    protectPaths: ["F:/tmp/protected"],
  })
  await plugin.setup(ctx)

  const history = [
    {
      id: "a1",
      content: [{ type: "text", text: "x".repeat(100_000) }],
    },
    {
      id: "a2",
      content: [
        {
          type: "tool-call",
          id: "c1",
          name: "write",
          input: { filePath: "F:/tmp/normal.ts", content: "y".repeat(60_000) },
        },
        { type: "tool-result", id: "c1", name: "write", result: { type: "text", value: "ok" } },
      ],
    },
    {
      id: "a3",
      content: [
        {
          type: "tool-call",
          id: "c2",
          name: "write",
          input: { filePath: "F:/tmp/protected/index.ts", content: "z".repeat(50_000) },
        },
        { type: "tool-result", id: "c2", name: "write", result: { type: "text", value: "ok" } },
      ],
    },
    { id: "a4", content: [{ type: "text", text: "one" }] },
    {
      id: "a5",
      content: [
        {
          type: "tool-call",
          id: "c3",
          name: "write",
          input: { filePath: "F:/tmp/newer.ts", content: "w".repeat(40_000) },
        },
        { type: "tool-result", id: "c3", name: "write", result: { type: "text", value: "ok" } },
        { type: "text", text: "two" },
      ],
    },
    { id: "a6", content: [{ type: "text", text: "three" }] },
  ]

  const linesAfter = async () => {
    await runContext(ctx, hooks, "count", structuredClone(history))
    await commands[0].execute({ sessionID: "count" })
    const report = String(synthetic.at(-1).description).match(/full report: ([^)\s]+)/)?.[1]
    assert.ok(report, "the report must have a full report file")
    const text = readFileSync(report, "utf8")
    const pick = (label) => text.split("\n").find((l) => l.includes(label))
    return {
      calls: pick("call arguments:"),
      text: pick("own text:"),
      left: pick("left alone:"),
    }
  }

  const first = await linesAfter()
  // Sanity: all three counters are non-zero, or the test proves nothing.
  assert.ok(/call arguments:\s+[1-9]/.test(first.calls), `calls must be counted:\n${first.calls}`)
  assert.ok(/own text:\s+[1-9]/.test(first.text), `text must be counted:\n${first.text}`)
  assert.ok(/left alone:\s+[1-9]/.test(first.left), `protected parts must be counted:\n${first.left}`)

  // Same history, one more request. A part already counted must not be
  // counted again.
  const second = await linesAfter()
  assert.equal(second.calls, first.calls, "call arguments must be counted once")
  assert.equal(second.text, first.text, "own text must be counted once")
  assert.equal(second.left, first.left, "protected parts must be counted once")
}

// --- BUG 7: an old image is a file item, and no rule saw it -----------------
//
// `read` on a png returns `{type:"file", uri:"data:image/png;base64,..."}`, with
// no text at all. Every trimmer walked past it, so one session carried 47 MB of
// old screenshots and the plugin cut 2% of the payload.
{
  const dir = `${process.env.TEMP}\\slim-image-${Date.now()}`
  const tiny = { keepFullOutput: true, directory: dir, resultKeepBytes: 4_000, stairBytes: 1_000 }
  const { ctx, hooks } = mockCtx(tiny)
  await plugin.setup(ctx)

  const shot = (n) => ({
    type: "file",
    uri: `data:image/png;base64,${Buffer.from("p".repeat(30_000) + n).toString("base64")}`,
    mime: "image/png",
    name: `C:\\shots\\shot-${n}.png`,
  })
  const imageResult = (id, n) => ({
    id,
    content: [
      {
        type: "tool-result",
        id,
        name: "read",
        result: { type: "content", value: [shot(n)] },
      },
    ],
  })

  const e = await runContext(ctx, hooks, "images", [
    imageResult("prt_1", 1),
    imageResult("prt_2", 2),
    imageResult("prt_3", 3),
    { id: "m4", content: [{ type: "text", text: "done" }] },
  ])
  const first = e.messages[0].content[0].result.value[0]
  assert.equal(first.type, "text", "an old image must become a pointer")
  assert.ok(first.text.includes("shot-1.png"), `the pointer must name the source file:\n${first.text}`)
  assert.ok(/elided/.test(first.text), `the pointer must say it was elided:\n${first.text}`)
  // Not a text digest: restoring a png as utf8 would be corruption.
  assert.equal(markerPath(first.text), undefined, "an image pointer is not a restorable text marker")

  const last = e.messages[2].content[0].result.value[0]
  assert.equal(last.type, "file", "the newest image must stay whole")
  assert.ok(last.uri.startsWith("data:image/png"), "the newest image must keep its bytes")
}

console.log("all checks passed")
