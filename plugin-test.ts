/**
 * Integration test: drives the plugin's hooks with a mock context and checks
 * that the report it prints is honest.
 *
 * The unit tests check the helpers. This one checks the wiring: that the
 * hooks fire, that the payload really is smaller, and that `/slim-stats`
 * reports numbers that match what the plugin did.
 */
import assert from "node:assert/strict"
import { readFileSync } from "node:fs"
import plugin, { BRIEF as BRIEF_TEXT } from "./index.ts"

const BRIEF_SIZE = Buffer.byteLength(BRIEF_TEXT, "utf8")

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

/**
 * The full report, read back from the file the synthetic points at.
 *
 * The TUI renders a synthetic's description in a single-line text element, so
 * the description is the one-line summary and the full report is written to a
 * file. The tests that check the numbers need the whole thing.
 */
function fullReport(synthetic: any[]): string {
  const description = String(synthetic.at(-1)?.description ?? "")
  const file = description.match(/full report: ([^)\s]+)/)?.[1]
  if (file) return readFileSync(file, "utf8")
  // A session with no requests has nothing to write to a file, and the
  // description carries the message on its own.
  return description
}

/** The same measurement the plugin makes: system bytes plus the message parts. */
function partBytes(part: any): number {
  if (!part || typeof part !== "object") return 0
  if (part.type === "reasoning") return bytes(part.text ?? "")
  if (part.type === "text") return bytes(part.text ?? "")
  if (part.type === "tool-result") {
    const r = part.result
    if (!r) return 0
    if (r.type === "text") return bytes(r.value ?? "")
    if (r.type === "content" && Array.isArray(r.value))
      return r.value.reduce((n: number, i: any) => n + (i?.type === "text" ? bytes(i.text) : 0), 0)
    return 0
  }
  return 0
}

const payloadOf = (system: any[], messages: any[], tools: any = {}) =>
  system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
  Object.values(tools).reduce(
    (n: number, t: any) => n + bytes(t?.description ?? "") + bytes(JSON.stringify(t?.input ?? {})),
    0,
  ) +
  messages.reduce((n: number, m: any) => n + (m.content ?? []).reduce((k: number, p: any) => k + partBytes(p), 0), 0)

/** Parses "29.8 KB" back into bytes. */
function parseSize(s: string): number {
  const m = s.trim().match(/^([\d,.]+)\s*(B|KB|MB)$/i)
  if (!m) throw new Error(`cannot parse size: ${s}`)
  const n = Number(m[1].replace(/,/g, ""))
  const unit = m[2].toUpperCase()
  return unit === "B" ? n : unit === "KB" ? n * 1024 : n * 1024 * 1024
}

/** Builds a mock ctx that records the hooks the plugin registers. */
function mockCtx(options: any = {}) {
  const hooks: Record<string, Function> = {}
  const commands: any[] = []
  const synthetic: any[] = []

  const ctx = {
    options,
    tool: {
      hook: async (name: string, fn: Function) => {
        hooks[`tool.${name}`] = fn
      },
    },
    session: {
      hook: async (name: string, fn: Function) => {
        hooks[`session.${name}`] = fn
      },
      synthetic: async (e: any) => {
        synthetic.push(e)
      },
    },
    command: {
      transform: async (fn: any) => {
        fn({ add: (c: any) => commands.push(c) })
      },
    },
  }
  return { ctx, hooks, commands, synthetic }
}

/** A tool result of `n` bytes. */
const bigResult = (n: number) => "x".repeat(n)

async function run() {
  const { ctx, hooks, commands, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  assert.ok(hooks["tool.execute.after"], "the tool hook is registered")
  assert.ok(hooks["session.context"], "the context hook is registered")
  assert.equal(commands.length, 1, "one command is registered")
  assert.equal(commands[0].name, "slim-stats")

  // --- A big tool result arrives. ------------------------------------------
  const first = {
    sessionID: "ses_test",
    tool: "read",
    status: "completed",
    result: { content: [{ type: "text", text: bigResult(50_000) }] },
  }
  await hooks["tool.execute.after"](first)

  // The result must have been digested at source, so this is what persists.
  const text = first.result.content[0].text
  assert.ok(text.length < 50_000, "the result must be digested at source")
  assert.ok(text.includes("[slim:"), "the digest must be marked")
  assert.ok(!text.includes("x".repeat(50_000)), "the raw result must be gone")

  // The same result a second time is a duplicate, not a second digest.
  const second = {
    sessionID: "ses_test",
    tool: "read",
    status: "completed",
    result: { content: [{ type: "text", text: bigResult(50_000) }] },
  }
  await hooks["tool.execute.after"](second)
  assert.ok(second.result.content[0].text.includes("duplicate"), "a repeated result must be collapsed")

  // --- A request goes out with a history that holds the big result. ---------
  const historyPart = { type: "tool-result", id: "prt_big", name: "read", result: { type: "text", value: bigResult(50_000) } }
  // An OLD result, and then a newer one. The old result is out of the keep
  // band, so it is elided; the newer one is the model's current context, so
  // it is kept whole.
  const oldPart = { type: "tool-result", id: "prt_old", name: "read", result: { type: "text", value: bigResult(50_000) } }
  const messages = Array.from({ length: 20 }, (_, i) => ({
    id: `m${i}`,
    content: [{ type: "reasoning", text: "r".repeat(5_000) }],
  }))
  messages[17].content = [oldPart]
  messages[19].content = [historyPart]
  const system = [{ type: "text", text: "s".repeat(40_000) }]
  const tools = { read: { description: "d".repeat(3_000), input: { type: "object", properties: { path: { type: "string", description: "p".repeat(2_000) } } } } }

  let before = 0
  let after = 0
  const reports: string[] = []
  for (let i = 0; i < 3; i++) {
    const event = {
      sessionID: "ses_test",
      system: system.map((p) => ({ ...p })),
      tools: JSON.parse(JSON.stringify(tools)),
      messages: messages.map((m) => ({ ...m, content: [...m.content] })),
    }
    // Measure the untrimmed payload ourselves, the way the plugin does.
    before = payloadOf(event.system, event.messages, event.tools)
    await hooks["session.context"](event)
    // The plugin adds a fixed behaviour brief to every request. The report
    // measures the payload before that brief, so the brief is excluded here
    // too — otherwise the two would differ by exactly the brief's size.
    const brief = event.system.find((p: any) => (p.text ?? "").startsWith("## Context discipline"))
    const briefBytes = brief ? bytes(brief.text ?? "") : 0
    after = payloadOf(event.system, event.messages, event.tools) - briefBytes
    await commands[0].execute({ sessionID: "ses_test" })
    reports.push(fullReport(synthetic))
  }

  assert.ok(after < before, `the payload must shrink: ${after} !< ${before}`)

  // --- The report must agree with what actually happened. -------------------
  const report = reports.at(-1)!
  assert.ok(report.includes("requests: 3"), `the request count must be right:\n${report}`)

  // The reported "after" must match the payload we measured.
  const afterLine = report.match(/This request[\s\S]*?^\s+(.+?)\s+\(\d+% smaller/m)
  assert.ok(afterLine, `the report must state the payload size:\n${report}`)
  const reported = parseSize(afterLine[1])
  // The report rounds to one decimal, so allow that much.
  assert.ok(Math.abs(reported - after) / Math.max(1, after) < 0.01, `the reported size must match the measured size: ${reported} vs ${after}`)

  // The cumulative tool-result saving must not grow between requests. The
  // history is re-elided on every request, so counting it again each time
  // would inflate the saving by the number of requests.
  const savedOf = (r: string) => parseSize(r.match(/tool results:\s+([\d.,]+\s*\w+)/)![1])
  assert.equal(savedOf(reports[0]), savedOf(reports[2]), "the tool result saving must not grow between requests")
  assert.equal(savedOf(reports[1]), savedOf(reports[2]), "the tool result saving must not grow between requests")

  // Two 50 KB results were trimmed: one at source, one in the history. The
  // saving must be under the raw bytes and over most of them.
  const resultSaved = savedOf(report)
  assert.ok(resultSaved < 100_000, `the saving must be under the raw bytes: ${resultSaved}`)
  assert.ok(resultSaved > 80_000, `the saving must be most of the raw bytes: ${resultSaved}`)
  // The reasoning saving must be the pruned messages, counted once.
  const reasoningSaved = parseSize(report.match(/reasoning:\s+([\d.,]+\s*\w+)/)![1])
  // 20 messages of 5 KB each is 100 KB of reasoning. The budget is 120 KB,
  // so the byte budget keeps all of it — and the floor keeps the newest
  // 40 KB whole whatever the budget says. Nothing is pruned, and the report
  // must say so rather than claim a saving.
  const expectedReasoning = 0
  assert.ok(
    Math.abs(reasoningSaved - expectedReasoning) / Math.max(1, expectedReasoning) < 0.01,
    `the reasoning saving must be the pruned messages: ${reasoningSaved} vs ${expectedReasoning}`,
  )
  assert.ok(report.includes("(0 parts)"), `the part count must be right:\n${report}`)

  // --- A session with no requests must say so. ------------------------------
  await commands[0].execute({ sessionID: "ses_other" })
  assert.ok(fullReport(synthetic).includes("no requests recorded yet"))
}

/**
 * The reasoning budget must actually bite.
 *
 * The test above uses 100 KB of reasoning, which is inside the 120 KB budget,
 * so nothing is pruned. This one uses 1 MB, and the saving must be most of
 * it — and the newest 40 KB must survive whole, because that is the model's
 * current train of thought.
 */
{
  const { ctx, hooks } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  const messages = Array.from({ length: 200 }, (_, i) => ({
    id: `m${i}`,
    content: [{ type: "reasoning", text: "r".repeat(5_000) }],
  }))
  const event = {
    sessionID: "ses_budget",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: messages.map((m) => ({ ...m, content: [...m.content] })),
  }
  await hooks["session.context"](event)

  const kept = event.messages.flatMap((m: any) => m.content).filter((p: any) => p.type === "reasoning")
  const bytesKept = kept.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0)
  // 1 MB in, at most 120 KB of budget plus the 40 KB floor out.
  assert.ok(kept.length < 200, `reasoning must be pruned: ${kept.length} parts kept`)
  assert.ok(bytesKept <= 120_000 + 40_000, `the budget must be respected: ${bytesKept} bytes kept`)
  assert.ok(bytesKept >= 40_000, `the newest reasoning must be kept whole: ${bytesKept} bytes kept`)
  // The newest part must be untouched.
  assert.equal(
    kept[kept.length - 1].text,
    "r".repeat(5_000),
    "the newest reasoning must be kept whole",
  )
}

/**
 * The report must never wake the model.
 *
 * The bug: `synthetic()` without `resume: false` calls `execution.wake()`.
 * The numbers appeared in the session and then the model was asked to talk
 * about them, costing a whole turn for something the user already sees.
 */
{
  const { ctx, hooks, commands, synthetic } = mockCtx({ keepFullOutput: false })
  const wakes: any[] = []
  ctx.session.synthetic = async (e: any) => {
    synthetic.push(e)
    if (e.resume !== false) wakes.push(e)
  }
  await (plugin as any).setup(ctx)

  // A request, so there is something to report.
  await hooks["session.context"]({
    sessionID: "ses_wake",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "hello" }] }],
  })

  await commands[0].execute({ sessionID: "ses_wake" })
  assert.equal(wakes.length, 0, "the command must not wake the model")
  assert.equal(synthetic.at(-1).resume, false, "the synthetic must be marked resume: false")
  // The TUI renders a synthetic's description, not its body, and skips a
  // synthetic that has no description at all.
  assert.ok(synthetic.at(-1).description, "the synthetic must carry a description, or the TUI hides it")
  assert.ok(
    String(synthetic.at(-1).description).includes("Token slim"),
    "the description must hold the report, because that is what the TUI shows",
  )
  // The body is what enters the model's context, so it must stay short.
  assert.ok(
    bytes(String(synthetic.at(-1).text ?? "")) < 200,
    `the body must stay short, it enters the model context: ${synthetic.at(-1).text}`,
  )
}

/**
 * The CLI does not dispatch registered commands, it sends them as a plain
 * prompt. The prompt hook must catch that and put the numbers in front of
 * the model, so the user gets the report instead of the model guessing what
 * "/slim-stats" means.
 */
{
  const { ctx, hooks, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  await hooks["session.context"]({
    sessionID: "ses_cli",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "hello" }] }],
  })

  const event = { sessionID: "ses_cli", prompt: { text: "/slim-stats" } }
  await hooks["session.prompt"](event)

  assert.ok(event.prompt.text.includes("Token slim"), "the report must replace the prompt")
  assert.ok(event.prompt.text.includes("requests:"), "the report must have the numbers")
  assert.ok(!event.prompt.text.startsWith("/slim-stats"), "the slash command must be gone")
  // The model must not be woken twice: no synthetic is needed here.
  assert.equal(synthetic.length, 0, "the CLI path must not add a synthetic")
}

/**
 * A prompt that merely mentions the command must not be swallowed.
 */
{
  const { ctx, hooks, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  const event = { sessionID: "ses_x", prompt: { text: "run /slim-stats for me and explain it" } }
  await hooks["session.prompt"](event)
  assert.equal(synthetic.length, 0, "a prompt that only mentions the command must be left alone")
  assert.equal(event.prompt.text, "run /slim-stats for me and explain it")
}

/**
 * The system prompt cap must trim TO the budget, not past it.
 *
 * The bug: the cap passed the excess (how many bytes had to go) to the
 * trimming function, which treats its argument as how many bytes to KEEP.
 * A 34 KB prompt with a 23 KB budget ended up at 15 KB, throwing away 7 KB
 * of system prompt the cap never asked to remove.
 */
{
  const { ctx, hooks, commands, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  const systemBytes = 24000
  const big = "A".repeat(20000)
  const rest = "B".repeat(14918)
  const event = {
    sessionID: "ses_cap",
    system: [
      { type: "text", text: big },
      { type: "text", text: rest },
    ],
    tools: {},
    messages: [],
  }
  await hooks["session.context"](event)

  const total = event.system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0)
  // The brief is added after the cap, so the cap budget excludes it.
  const budget = systemBytes - bytes("## Context discipline\n".padEnd(0, "x"))
  assert.ok(total <= systemBytes, `the system prompt must be under the cap: ${total}`)

  // The cap must not throw away more than it has to. With a 34.1 KB prompt
  // and a 23 KB budget, the result must be at least 22 KB.
  assert.ok(total >= 22_000, `the cap must not over-trim: ${total} is well under the budget`)

  await commands[0].execute({ sessionID: "ses_cap" })
  const report = fullReport(synthetic)
  const dropped = parseSize(report.match(/system prompt:\s+([\d.,]+\s*\w+)/)![1])
  // The cap has to make room for the brief, so the saving is the excess over
  // the cap MINUS the brief. What matters is that the number reported is the
  // number that was really dropped, not that it is under a fixed figure.
  const excess = big.length + rest.length - (systemBytes - BRIEF_SIZE)
  assert.ok(
    Math.abs(dropped - excess) < 200,
    `the reported system saving must be the real one: ${dropped} vs ${excess}`,
  )
}

/**
 * The payload the report talks about must include the tool schemas.
 *
 * They are sent with every request, so leaving them out understates what
 * went to the model.
 */
{
  const { ctx, hooks, commands, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  const toolBytes = 12_000
  const tools = {
    read: {
      description: "d".repeat(6_000),
      input: { type: "object", properties: { path: { type: "string", description: "p".repeat(6_000) } } },
    },
  }
  const event = {
    sessionID: "ses_tools",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: JSON.parse(JSON.stringify(tools)),
    messages: [{ id: "m0", content: [{ type: "text", text: "hello" }] }],
  }
  await hooks["session.context"](event)

  await commands[0].execute({ sessionID: "ses_tools" })
  const report = fullReport(synthetic)
  const afterLine = report.match(/This request[\s\S]*?^\s+(.+?)\s+\(\d+% smaller/m)
  assert.ok(afterLine, `the report must state the payload size:\n${report}`)
  const reported = parseSize(afterLine[1])

  // The schemas are capped, so the payload must be bigger than the messages
  // and the system prompt alone.
  const systemOnly = event.system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0)
  assert.ok(reported > systemOnly, `the payload must include the tool schemas: ${reported} vs ${systemOnly}`)
  assert.ok(reported < toolBytes, `the schemas must be capped, not sent whole: ${reported}`)
}

/**
 * The peak line must compare the largest request before and after, not the
 * largest untrimmed payload against the most recent trimmed one.
 */
{
  const { ctx, hooks, commands, synthetic } = mockCtx({ keepFullOutput: false })
  await (plugin as any).setup(ctx)

  // First request: small.
  await hooks["session.context"]({
    sessionID: "ses_peak",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "a" }] }],
  })
  // Second request: much bigger, so it becomes the peak.
  const big = "r".repeat(50_000)
  await hooks["session.context"]({
    sessionID: "ses_peak",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: Array.from({ length: 20 }, (_, i) => ({ id: `m${i}`, content: [{ type: "reasoning", text: big }] })),
  })
  // Third request: tiny. It must not become the peak, and it must not be
  // compared against the peak's untrimmed size.
  await hooks["session.context"]({
    sessionID: "ses_peak",
    system: [{ type: "text", text: "s".repeat(1_000) }],
    tools: {},
    messages: [{ id: "m0", content: [{ type: "text", text: "a" }] }],
  })

  await commands[0].execute({ sessionID: "ses_peak" })
  const report = fullReport(synthetic)
  const peakLine = report.match(/Largest request this session, before and after:\s*\n\s*(.+)/)
  assert.ok(peakLine, `the report must state the peak before and after:\n${report}`)
  // Both numbers must be on the line, so the saving is a real comparison.
  assert.ok(peakLine[1].includes("->"), `the peak line must show both sizes:\n${report}`)
  const [beforeStr, afterStr] = peakLine[1].split("->").map((s: string) => s.trim())
  const peakBefore = parseSize(beforeStr.split("  ")[0])
  const peakAfter = parseSize(afterStr.split("  ")[0])
  assert.ok(peakBefore > peakAfter, `the peak must show a saving: ${peakBefore} -> ${peakAfter}`)
  // The peak's trimmed size must be the trimmed size of the SAME request,
  // not of the tiny third request.
  assert.ok(peakAfter > 1_000, `the peak must compare like with like: ${peakAfter}`)
}

await run()
console.log("all checks passed")
