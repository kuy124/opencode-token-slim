/**
 * Does the trimming cost the model anything?
 *
 * A reduction is only worth having if the model can still do its job. This
 * checks the three things the model actually needs from a trimmed history:
 *
 *   1. The current working context is intact. The newest reasoning, the
 *      newest results and the newest calls are kept WHOLE, not digested.
 *   2. Nothing is silently lost. Every elided result names a file the model
 *      can re-read, and the file exists and holds the original text.
 *   3. The elided form is readable. A marker says what it is, in one line,
 *      in the same shape as the content around it.
 */
import assert from "node:assert/strict"
import { createHash } from "node:crypto"
import { existsSync, readFileSync } from "node:fs"
import plugin from "./index.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

function mockCtx(options: any) {
  const hooks: Record<string, Function> = {}
  const synthetic: any[] = []
  const commands: any[] = []
  const ctx = {
    options,
    tool: { hook: async (n: string, f: Function) => { hooks[`tool.${n}`] = f } },
    session: {
      hook: async (n: string, f: Function) => { hooks[`session.${n}`] = f },
      synthetic: async (e: any) => { synthetic.push(e) },
    },
    command: { transform: async (f: Function) => { f({ add: (c: any) => commands.push(c) }) } },
  }
  return { ctx, hooks, synthetic, commands }
}

const TMP = `${process.env.TEMP}\\slim-quality`
const OPTS = { keepFullOutput: true, directory: TMP }

function build(nMessages: number, resultBytes: number) {
  const messages: any[] = []
  for (let i = 0; i < nMessages; i++) {
    messages.push({
      id: `m${i}`,
      role: "assistant",
      content: [
        { type: "reasoning", text: `reasoning-${i}-`.padEnd(400, "r") },
        { type: "tool-call", id: `c${i}`, name: "write", input: { path: `f${i}.txt`, content: "x".padEnd(3000, "w") } },
        { type: "tool-result", id: `c${i}`, name: "read", result: { type: "text", value: "RESULT".padEnd(resultBytes, "y") } },
        { type: "text", text: `answer ${i}` },
      ],
    })
  }
  return messages
}

// --- 1. The newest content is kept WHOLE ------------------------------------
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const messages = build(60, 8000)
  const event = {
    sessionID: "q1",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  const last = event.messages[event.messages.length - 1]
  const reasoning = last.content.find((p: any) => p.type === "reasoning")
  const result = last.content.find((p: any) => p.type === "tool-result")
  assert.equal(reasoning.text, `reasoning-59-`.padEnd(400, "r"), "the newest reasoning must be kept whole")
  assert.equal(result.result.value.length, 8000, "the newest result must be kept whole")
}

// --- 2. Nothing is lost: every elided result names a real file --------------
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const messages = build(60, 8000)
  const event = {
    sessionID: "q2",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  let checked = 0
  for (const m of event.messages) {
    for (const p of m.content) {
      if (p.type !== "tool-result") continue
      const value = p.result.value
      if (typeof value !== "string") continue
      if (!value.startsWith("[slim:")) continue
      // An elided result must name a file that exists and holds the original.
      const match = /Full text: (\S+?)\]/.exec(value)
      assert.ok(match, `an elided result must name its file: ${value}`)
      const file = match[1]
      assert.ok(existsSync(file), `the named file must exist: ${file}`)
      const original = readFileSync(file, "utf8")
      assert.ok(original.startsWith("RESULT"), `the file must hold the original text: ${file}`)
      assert.equal(original.length, 8000, `the file must be the whole result: ${file}`)
      checked++
    }
  }
  assert.ok(checked > 0, "the test must have actually elided something")
}

// --- 3. The marker is one line, and says what it is -------------------------
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const messages = build(60, 8000)
  const event = {
    sessionID: "q3",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  for (const m of event.messages) {
    for (const p of m.content) {
      if (p.type !== "tool-result") continue
      const value = p.result.value
      if (typeof value !== "string") continue
      if (!value.startsWith("[slim:")) continue
      assert.ok(!value.includes("\n"), `the marker must be one line: ${value}`)
      // It must name the tool and the size, so the model knows what it lost.
      assert.ok(/read result/.test(value), `the marker must name the tool: ${value}`)
      assert.ok(/KB/.test(value), `the marker must state the size: ${value}`)
    }
  }
}

// --- 4. A tool CALL is never dropped, only its arguments are trimmed --------
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const messages = build(60, 8000)
  const event = {
    sessionID: "q4",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  for (const m of event.messages) {
    const call = m.content.find((p: any) => p.type === "tool-call")
    assert.ok(call, "every tool call must survive")
    assert.ok(call.input.path, `the call must keep its shape: ${JSON.stringify(call.input).slice(0, 200)}`)
  }
}

// --- 5. a tool CALL's arguments are shortened, and keep their shape ----------
//
// A `write` carries the whole file in its arguments, so a session that writes
// ten files replays all ten files on every turn forever. The call itself has
// to survive — the model needs to know it made the call — and the shape of
// the arguments has to survive too, or the call no longer means anything.
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const messages = Array.from({ length: 60 }, (_, i) => ({
    id: `m${i}`,
    content: [
      {
        type: "tool-call",
        id: `c${i}`,
        name: "write",
        input: { path: `f${i}.txt`, content: "w".repeat(20_000) },
      },
      { type: "tool-result", id: `c${i}`, name: "write", result: { type: "text", value: "ok" } },
    ],
  }))
  const event = {
    sessionID: "q5",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  let trimmed = 0
  for (const m of event.messages) {
    const call = m.content.find((p: any) => p.type === "tool-call")
    assert.ok(call, "every tool call must survive")
    assert.equal(typeof call.input.path, "string", "the path must survive")
    assert.equal(typeof call.input.content, "string", "the content must survive as a string")
    if (call.input.content.length < 20_000) trimmed++
  }
  assert.ok(trimmed > 0, "the old calls must have their arguments shortened")
}

// --- 6. the request is byte-identical between steps --------------------------
//
// The provider's prompt cache only helps if the request does not change. The
// cut is a byte position measured from the start of the history, rounded down
// to a step, so a little new content must NOT move it: the request that goes
// out has to be exactly the same bytes as the one before.
//
// A sliding window fails this on every turn, and the whole history is
// re-read at full price.
{
  const { ctx, hooks } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  const sent = (messages: any[]) => {
    const event = {
      sessionID: "q6",
      system: [{ type: "text", text: "s".repeat(1000) }],
      tools: {},
      messages: messages.map((m) => ({ ...m, content: [...m.content] })),
    }
    return event
  }
  // A hash, so a failure prints a short message instead of a megabyte of
  // repeated characters.
  const digest = (event: any, upto: number) =>
    createHash("sha1").update(JSON.stringify(event.messages.slice(0, upto))).digest("hex")

  const history = Array.from({ length: 200 }, (_, i) => ({
    id: `m${i}`,
    content: [{ type: "reasoning", text: "r".repeat(5_000) }],
  }))

  const first = sent(history)
  await hooks["session.context"](first)
  const firstHash = digest(first, 200)

  // The same history again: the request must be identical.
  const second = sent(history)
  await hooks["session.context"](second)
  assert.equal(digest(second, 200), firstHash, "the same history must produce the same request")

  // A little new content: the cut has not moved, so the OLD part of the
  // request must be identical. The tail grows, which is the point.
  const grown = [...history, { id: "new", content: [{ type: "reasoning", text: "n".repeat(500) }] }]
  const third = sent(grown)
  await hooks["session.context"](third)
  assert.equal(
    digest(third, 200),
    firstHash,
    "the cut must not move until a whole step has accumulated",
  )
}

console.log("all checks passed")
