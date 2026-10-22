/**
 * The report's own numbers must be honest.
 *
 * The bug this guards: the behaviour brief is ADDED to every request. It was
 * measured into the "after" number but not into the "before" one, so the
 * report showed the payload GROWING on every request — and a session that
 * trimmed nothing looked like it had lost 20% to the brief.
 *
 * The plugin's internal numbers are checked here, not a re-implementation of
 * them: the report is read back from the synthetic the plugin produced.
 */
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import plugin from "./index.ts"

function mockCtx(options: any) {
  const hooks: Record<string, Function> = {}
  const synthetic: any[] = []
  const commands: any[] = []
  const ctx = {
    options,
    tool: { hook: async (name: string, fn: Function) => { hooks[`tool.${name}`] = fn } },
    session: {
      hook: async (name: string, fn: Function) => { hooks[`session.${name}`] = fn },
      synthetic: async (e: any) => { synthetic.push(e) },
    },
    command: { transform: async (fn: Function) => { fn({ add: (c: any) => commands.push(c) }) } },
  }
  return { ctx, hooks, synthetic, commands }
}

const OPTS = { keepFullOutput: false }

// --- 1. a session that trims nothing must not look like it lost bytes ------
{
  const { ctx, hooks, synthetic, commands } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  await hooks["session.context"]({
    sessionID: "s1",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "hello" }] }],
  })

  await commands[0].execute({ sessionID: "s1" })
  const description = String(synthetic.at(-1).description)

  // The percentage is the thing the user reads, and it must not be negative
  // when nothing was trimmed.
  const pct = Number(description.match(/(\d+)% smaller than the untrimmed history/)?.[1] ?? -999)
  assert.ok(pct >= 0, `a session that trims nothing must not report a loss: ${pct}%\n${description}`)

  // The full report is in the file the description points at. Nothing was
  // trimmed here, so the brief was not added either — and the report must not
  // claim it was, because a request that trims nothing must not grow by 800
  // bytes of advice.
  const file = description.match(/full report: ([^)\s]+)/)?.[1]
  assert.ok(file, `the description must point at the full report:\n${description}`)
  const full = readFileSync(file!, "utf8")
  assert.ok(!/brief added:/.test(full), `a request that trims nothing must not pay for the brief:\n${full}`)
  assert.ok(
    !/Actually sent, including the brief:/.test(full),
    `the report must not describe a brief that was not added:\n${full}`,
  )
  assert.ok(
    /This request \(trimmed, before the brief is added\):/.test(full),
    `the full report must state the trimmed size:\n${full}`,
  )
}

// --- 2. a session that DOES trim must report a real saving ----------------
{
  const { ctx, hooks, synthetic, commands } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  // The newest result is kept whole — it is the model's current context — so
  // the one that gets elided has to be an older one.
  await hooks["session.context"]({
    sessionID: "s2",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: [
      { id: "m0", content: [{ type: "text", text: "hello" }] },
      {
        id: "m1",
        content: [{ type: "tool-result", id: "t1", name: "read", result: { type: "text", value: "x".repeat(50_000) } }],
      },
      {
        id: "m2",
        content: [{ type: "tool-result", id: "t2", name: "read", result: { type: "text", value: "y".repeat(50_000) } }],
      },
    ],
  })

  await commands[0].execute({ sessionID: "s2" })
  const report = String(synthetic.at(-1).description)
  const pct = Number(report.match(/(\d+)% smaller than the untrimmed history/)?.[1] ?? 0)
  // The newest 50 KB result is kept whole on purpose, so the saving is one
  // result out of two, not two out of two. It must still be a real saving,
  // and it must be reported.
  assert.ok(pct > 40, `a 50 KB result must show a real saving: ${pct}%\n${report}`)
  assert.ok(
    /tool results:\s+[\d.]+\s*[KM]?B/.test(report),
    `the tool result saving must be reported:\n${report}`,
  )

  // Something WAS trimmed, so the brief is added and has to be reported on its
  // own line: it is a cost, and a report that hides it is a lie.
  const file = report.match(/full report: ([^)\s]+)/)?.[1]
  assert.ok(file, `the description must point at the full report:\n${report}`)
  const full = readFileSync(file!, "utf8")
  assert.ok(/brief added:\s+\S+/.test(full), `a trimmed request must report the brief:\n${full}`)
  assert.ok(/Actually sent, including the brief:/.test(full), `the report must state what was sent:\n${full}`)
}

// --- 3. the TUI can only show ONE line, so the description must be one line --
{
  const { ctx, hooks, synthetic, commands } = mockCtx(OPTS)
  await (plugin as any).setup(ctx)

  await hooks["session.context"]({
    sessionID: "s3",
    system: [{ type: "text", text: "s".repeat(1000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "hello" }] }],
  })

  await commands[0].execute({ sessionID: "s3" })
  const description = String(synthetic.at(-1).description)
  assert.ok(
    !description.includes("\n"),
    `the description must be ONE line: the TUI renders it in a single <text> element.\n${description}`,
  )
  // It must still say something useful.
  assert.ok(description.includes("Token slim"), `the description must identify the report:\n${description}`)
  // And it must point at the full report.
  assert.ok(/full report: \S+/.test(description), `the description must point at the full report:\n${description}`)
  // The body still enters the model's context, so it stays short.
  assert.ok(
    Buffer.byteLength(String(synthetic.at(-1).text ?? ""), "utf8") < 200,
    "the body must stay short",
  )
}

console.log("all checks passed")
