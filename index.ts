/**
 * opencode-token-slim
 *
 * Cuts the per-request payload an OpenCode agent sends to its model.
 *
 * Measured on a real session (846 messages, 5.2 MB request payload):
 *   reasoning     2.37 MB (46%)
 *   tool results  2.06 MB (40%)
 *   tool calls    0.62 MB (12%)
 *   own text      0.09 MB (2%)
 *
 * Every request re-sends that payload, so a long session pays for it again on
 * every turn.
 *
 * The whole plugin is ONE rule applied to four kinds of content. Each kind
 * gets a byte budget: the newest end of it is kept whole, the next band is
 * shortened, and the rest is replaced. A byte budget, not a window of N
 * messages — a session's content is wildly uneven, so a message count keeps a
 * wildly different amount from one session to the next, and the saving swings
 * from 5% to 49% from turn to turn. A byte budget keeps the same amount
 * whatever produced it.
 *
 *   1. Reasoning is dropped past its budget. The cut is a byte position
 *      measured from the START of the history, rounded down to a step, so it
 *      only MOVES when a whole step has accumulated: between moves the
 *      request is byte-identical and the provider's prompt cache still hits.
 *      The newest reasoning is never cut — it is the model's current train of
 *      thought. A part carrying a provider signature is never touched:
 *      Anthropic-style thinking blocks must be replayed byte-identical.
 *   2. Tool results past their budget become a pointer to the full text on
 *      disk, and the band just below that keeps a head+tail digest. The
 *      digest keeps the head and the tail as *contiguous line ranges* and
 *      names the omitted range, so the result never looks like a file with a
 *      corrupt region. The first and last line are always kept - clipped to
 *      the budget if they are too big - because a digest that starts or ends
 *      mid-file leaves the model with nothing to orient itself by. Nothing
 *      is ever replaced by something bigger: the payload must never grow.
 *   3. Tool-call arguments and the assistant's own text are shortened the
 *      same way. A `write` carries the whole file in its arguments, and a
 *      session that writes ten files replays all ten files forever.
 *   4. The system prompt and tool schemas are capped at safe boundaries, plus
 *      a short behaviour brief so the model stops re-reading and re-running
 *      the same things.
 *
 * It also fixes two things it does not shrink: provider item ids, which a
 * multi-resource gateway rejects on replay (see `stripItemIds`), and the
 * OpenCode-side truncation that would otherwise replace the digest with a
 * head-only cut (see `toolOutput`).
 *
 * Savings are measured per request and reported by `/slim-stats`.
 *
 * ## Self-protection
 *
 * A plugin that trims the agent's own context can trim the context the agent
 * is using to edit *this plugin*, which is how a session ends up writing a
 * broken edit back over a file it only half read. Two guards prevent that:
 *
 *   - `protectPaths` - file paths whose tool results are never trimmed. The
 *     plugin's own source directory is protected by default, so editing the
 *     plugin never reads a digest of the plugin. The ARGUMENTS of a call
 *     about a protected path are protected too, because trimming them would
 *     truncate the very edit being made.
 *   - `protectTools` - tool names whose results are never trimmed, for tools
 *     whose output the agent must read in full to act correctly.
 *
 * Both default to protecting this plugin's own directory, so the dangerous
 * case is covered without configuration.
 */

import fs from "node:fs"
import os from "node:os"
import path from "node:path"
import { createHash } from "node:crypto"
import { StringDecoder } from "node:string_decoder"
import { spawn } from "node:child_process"
import { fileURLToPath } from "node:url"

// ---------------------------------------------------------------------------
// Config
// ---------------------------------------------------------------------------

export interface SlimOptions {
  /** Tool-result text above this many bytes is replaced by a digest. */
  resultBytes: number
  /** Bytes kept from the head of an elided tool result. */
  headBytes: number
  /** Bytes kept from the tail of an elided tool result. */
  tailBytes: number
  /**
   * The reasoning kept whole, in bytes.
   *
   * This is a byte budget, not a message count. A session's reasoning is
   * wildly uneven: the median part is 1.5 KB and the largest is 70 KB, and
   * there is exactly one reasoning part per assistant message. A message-count
   * window therefore keeps either far too much or far too little, and the
   * saving swings from turn to turn. A byte budget keeps the same amount of
   * reasoning no matter how it is distributed.
   */
  reasoningBytes: number
  /**
   * The most recent reasoning kept whole whatever the budget says, in bytes.
   *
   * This is the "hot" reasoning: the model's current train of thought.
   * Cutting the reasoning the model is mid-way through is what makes a model
   * start repeating itself, so this floor is a hard one.
   */
  reasoningKeepBytes: number
  /**
   * The step used for every kind of content.
   *
   * This is how much has to accumulate before the cut moves. Between moves,
   * the request is byte-identical and the provider's prompt cache still
   * hits. A smaller step means a more even cut and more cache busts; a larger
   * step means a steadier cache and a lumpier cut.
   */
  stairBytes: number
  /**
   * The most recent tool results kept whole, in bytes.
   *
   * The same reasoning as the reasoning budget, applied to results: a result
   * the model is about to act on is worth its bytes, and one from fifty turns
   * ago is not.
   */
  resultKeepBytes: number
  /**
   * The band of tool results that get a digest rather than a pointer.
   *
   * The newest results are kept whole. The ones just behind them get a
   * digest — head and tail — because the model may still be working with
   * them. Everything older gets a one-line pointer, because a digest of
   * every result in a long session costs about half of what the results
   * themselves cost.
   */
  resultDigestBytes: number
  /**
   * The most recent tool-call arguments kept whole, in bytes.
   *
   * A tool call's arguments are replayed to the model every turn. The
   * arguments of a `write` are the whole file, so a session that writes ten
   * files replays all ten files forever. The calls themselves stay — the
   * model needs to know it made them — but the bulk of their arguments does
   * not.
   */
  callKeepBytes: number
  /**
   * The assistant's own text kept whole, in bytes.
   *
   * The model's own prose is the cheapest thing to keep and the most
   * expensive to re-derive: it is the record of what was decided and why.
   */
  textKeepBytes: number
  /**
   * The minimum size of a tool result before it is considered for elision.
   *
   * A result this small is cheaper to keep than to replace: the marker alone
   * is 150 bytes, and a head+tail digest of a 200 byte result is bigger than
   * the result. Below this size, nothing is gained.
   *
   * The floor is small on purpose. A session's tool results average about
   * 2.5 KB, so a floor of a few KB would exempt most of them and the
   * saving would depend on the session rather than on the rule.
   */
  elideFloorBytes: number
  /** Cap on the system prompt, in bytes. */
  systemBytes: number
  /** Cap on the combined tool schema bytes. */
  toolBytes: number
  /** Minimum size of a repeated result before it is collapsed. */
  duplicateBytes: number
  /**
   * Provider ids whose replayed provider item ids are dropped.
   *
   * The OpenAI Responses API replays the provider's own item ids
   * (`rs_...`, `msg_...`, `fc_...`) along with the history. Those ids are
   * scoped to the resource that created them, so a provider that
   * load-balances across resources rejects a replayed id outright. Only the
   * listed providers are affected; everything else is left alone.
   */
  stripItemIds: string[]
  /** Write the full text next to the digest instead of dropping it. */
  keepFullOutput: boolean
  /** Where full outputs are written. */
  directory: string
  /** Master switch. */
  enabled: boolean
  /**
   * Rewrite shell commands through `snip` before they run, so the tool result
   * that reaches the context is already filtered.
   *
   * This is the one optimisation that does not touch the model at all: snip
   * filters machine-generated noise (passing test names, progress bars, commit
   * metadata) out of the command's stdout. A `go test` that prints 18 package
   * lines becomes 16 tokens and still tells the model 1125 tests passed.
   */
  snip: boolean
  /** Path to the snip binary. Empty resolves it from PATH. */
  snipPath: string
  /** Milliseconds before a snip rewrite is abandoned and the command runs as-is. */
  snipTimeoutMs: number
  /**
   * Cap on the total bytes kept in the full-output directory.
   *
   * The directory grows without bound otherwise: every oversized result is
   * written there, and a long session writes a lot.
   */
  maxDiskBytes: number
  /** Where the per-session statistics are persisted, so they survive a restart. */
  statsDirectory: string
  /**
   * File paths whose tool results are never trimmed.
   *
   * A glob-ish match on the path: `*` matches within a segment, `**` matches
   * across segments. This is the guard that keeps the agent from reading a
   * digest of the file it is about to edit.
   */
  protectPaths: string[]
  /** Tool names whose results are never trimmed. */
  protectTools: string[]
  /**
   * OpenCode's own tool-output limit, in bytes.
   *
   * OpenCode truncates every tool result to this size on its own, keeping only
   * the head. The plugin runs in `tool.execute.after`, before that happens, so
   * its digest replaces the head-only cut and keeps the tail. The number is
   * needed to know how large a result OpenCode would have cut.
   */
  toolOutputBytes: number
}

const DEFAULTS: SlimOptions = {
  resultBytes: 12_000,
  headBytes: 3_000,
  tailBytes: 1_500,
  // The budgets below are sized against a real session: 670 LLM messages,
  // 4.8 MB of payload, of which 2.3 MB was reasoning and 1.9 MB was tool
  // results. They are byte budgets, not message counts, so the saving does
  // not swing with the shape of the session.
  //
  // They are also sized to hold 60% of the payload back at every session
  // length, not just the long ones. A budget that only bites in a long
  // session is the same bug as a message-count window, just smaller.
  reasoningBytes: 80_000,
  reasoningKeepBytes: 32_000,
  stairBytes: 16_000,
  resultKeepBytes: 32_000,
  resultDigestBytes: 48_000,
  callKeepBytes: 8_000,
  textKeepBytes: 16_000,
  elideFloorBytes: 512,
  systemBytes: 24_000,
  toolBytes: 10_000,
  duplicateBytes: 2_000,
  stripItemIds: [],
  keepFullOutput: true,
  directory: path.join(
    process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"),
    "opencode",
    "slim-output",
  ),
  enabled: true,
  snip: true,
  snipPath: "",
  snipTimeoutMs: 2_000,
  maxDiskBytes: 256 * 1024 * 1024,
  statsDirectory: path.join(
    process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"),
    "opencode",
    "slim-stats",
  ),
  protectPaths: [],
  protectTools: [],
  toolOutputBytes: 50 * 1024,
}

/**
 * The directory this plugin's own source lives in.
 *
 * It is protected by default: an agent editing the plugin must read the real
 * file, not a digest of it. Without this the plugin can trim its own source
 * out from under the agent that is trying to fix it, which is the one failure
 * mode that can leave a session unable to recover.
 */
const SELF_DIR = path.dirname(fileURLToPath(import.meta.url)).replace(/\\/g, "/").replace(/\/$/, "")

export function resolveOptions(raw: Record<string, any> | undefined): SlimOptions {
  const o = { ...DEFAULTS, ...(raw ?? {}) }
  return {
    resultBytes: positive(o.resultBytes, DEFAULTS.resultBytes),
    headBytes: positive(o.headBytes, DEFAULTS.headBytes),
    tailBytes: positive(o.tailBytes, DEFAULTS.tailBytes),
    reasoningBytes: positive(o.reasoningBytes, DEFAULTS.reasoningBytes),
    reasoningKeepBytes: positive(o.reasoningKeepBytes, DEFAULTS.reasoningKeepBytes),
    stairBytes: Math.max(1, Number(o.stairBytes ?? DEFAULTS.stairBytes) || 1),
    resultKeepBytes: positive(o.resultKeepBytes, DEFAULTS.resultKeepBytes),
    resultDigestBytes: positive(o.resultDigestBytes, DEFAULTS.resultDigestBytes),
    callKeepBytes: positive(o.callKeepBytes, DEFAULTS.callKeepBytes),
    textKeepBytes: positive(o.textKeepBytes, DEFAULTS.textKeepBytes),
    elideFloorBytes: positive(o.elideFloorBytes, DEFAULTS.elideFloorBytes),
    systemBytes: positive(o.systemBytes, DEFAULTS.systemBytes),
    toolBytes: positive(o.toolBytes, DEFAULTS.toolBytes),
    duplicateBytes: positive(o.duplicateBytes, DEFAULTS.duplicateBytes),
    stripItemIds: Array.isArray(o.stripItemIds) ? o.stripItemIds.filter((v: any) => typeof v === "string") : [],
    keepFullOutput: o.keepFullOutput !== false,
    directory: typeof o.directory === "string" && o.directory ? o.directory : DEFAULTS.directory,
    enabled: o.enabled !== false,
    snip: o.snip !== false,
    snipPath: typeof o.snipPath === "string" ? o.snipPath : DEFAULTS.snipPath,
    snipTimeoutMs: positive(o.snipTimeoutMs, DEFAULTS.snipTimeoutMs),
    maxDiskBytes: positive(o.maxDiskBytes, DEFAULTS.maxDiskBytes),
    statsDirectory:
      typeof o.statsDirectory === "string" && o.statsDirectory ? o.statsDirectory : DEFAULTS.statsDirectory,
    protectPaths: Array.isArray(o.protectPaths) ? o.protectPaths.filter((v: any) => typeof v === "string") : [],
    protectTools: Array.isArray(o.protectTools) ? o.protectTools.filter((v: any) => typeof v === "string") : [],
    toolOutputBytes: positive(o.toolOutputBytes, DEFAULTS.toolOutputBytes),
  }
}

function positive(value: any, fallback: number) {
  const n = Number(value)
  return Number.isFinite(n) && n > 0 ? n : fallback
}

// ---------------------------------------------------------------------------
// Byte helpers
// ---------------------------------------------------------------------------

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

/**
 * Cut a string to at most `max` bytes without leaving half a UTF-8
 * character behind. StringDecoder holds back an incomplete trailing
 * sequence instead of emitting a replacement character.
 */
const clipBytes = (s: string, max: number) => {
  if (bytes(s) <= max) return s
  return new StringDecoder("utf8").write(Buffer.from(s, "utf8").subarray(0, max))
}

/** Truncate on a byte budget without splitting a UTF-8 sequence. */
export function takeBytes(text: string, budget: number) {
  if (budget <= 0) return ""
  if (bytes(text) <= budget) return text
  const buf = Buffer.from(text, "utf8").subarray(0, budget)
  let end = buf.length
  while (end > 0 && (buf[end - 1] & 0xc0) === 0x80) end--
  if (end > 0 && (buf[end - 1] & 0x80) !== 0) end--
  return buf.subarray(0, end).toString("utf8")
}

/**
 * Keep the head and the tail of a piece of prose, and say what was cut.
 *
 * The model's own text is the record of what it decided and why. The head
 * says what it was doing; the tail says what it concluded. Taking only the
 * head throws the conclusion away, and the conclusion is usually the part
 * worth keeping.
 *
 * The marker is one line, so the model can see that something was removed
 * without having to parse a diff.
 */
export function trimText(text: string, budget: number): string {
  const size = bytes(text)
  if (budget <= 0 || size <= budget) return text
  const head = Math.floor(budget * 0.6)
  const tail = budget - head
  const marker = `\n\n… ${size - budget} bytes omitted …\n\n`
  const markerBytes = bytes(marker)
  const headText = takeBytes(text, Math.max(0, head - markerBytes))
  const tailText = takeTail(text, Math.max(0, tail - markerBytes))
  return headText + marker + tailText
}

/**
 * Take the last `budget` bytes of a string, without splitting a character.
 */
export function takeTail(text: string, budget: number): string {
  if (budget <= 0) return ""
  const buf = Buffer.from(text, "utf8")
  if (buf.length <= budget) return text
  let start = buf.length - budget
  // Move forward to the start of the character the cut landed in.
  while (start < buf.length && (buf[start] & 0xc0) === 0x80) start++
  if (start < buf.length && (buf[start] & 0x80) !== 0) start++
  return buf.subarray(start).toString("utf8")
}

// ---------------------------------------------------------------------------
// Path protection
// ---------------------------------------------------------------------------

/**
 * Match a path against a glob pattern.
 *
 * `*` matches within one path segment, `**` matches across segments. A
 * pattern that is a plain directory prefix also matches everything under it,
 * which is the common case ("protect this folder").
 */
export function matchesPattern(candidate: string, pattern: string): boolean {
  if (!pattern) return false
  const c = candidate.replace(/\\/g, "/")
  const p = pattern.replace(/\\/g, "/")
  if (c === p) return true
  // A plain path (no wildcard) protects the whole subtree.
  if (!/[*?[\]]/.test(p)) {
    return c === p || c.startsWith(p.endsWith("/") ? p : p + "/")
  }
  const rx = new RegExp(
    "^" +
      p
        .split("**")
        .map((part) =>
          part
            .split("*")
            .map((seg) => seg.replace(/[.+^${}()|[\]\\]/g, "\\$&"))
            .join("[^/]*"),
        )
        .join(".*") +
      "$",
  )
  return rx.test(c)
}

/**
 * Whether a tool result came from a path that must not be trimmed.
 *
 * The plugin's own directory is always protected, whatever the
 * configuration says. That is the guarantee that matters: a session editing
 * the plugin can always read the plugin.
 */
export function isProtectedPath(candidate: string, patterns: string[]): boolean {
  if (!candidate) return false
  if (matchesPattern(candidate, SELF_DIR)) return true
  for (const pattern of patterns) {
    if (matchesPattern(candidate, pattern)) return true
  }
  return false
}

/**
 * Pull a file path out of a tool result or tool call.
 *
 * The `read` tool reports the path in its output header, and the tool call
 * carries it in the input. Both are checked, because either one alone is
 * enough to identify the file and either one can be missing.
 */
export function extractPath(text: string): string | undefined {
  if (typeof text !== "string" || !text) return undefined
  // OpenCode's read tool prefixes its output with the path it read.
  const header = /^(?:Read|Edited|Wrote|Updated)\s+(?:file\s+)?["']?([^"'\n\r]+?)["']?(?:,|\s+lines\b|$)/im
  const m = header.exec(text)
  if (m) return m[1].trim()
  return undefined
}

/**
 * Pull a file path out of a tool call's arguments.
 *
 * The arguments are the input of whatever tool made the call, so the key
 * holding the path depends on the tool: `path` for read, `filePath` for
 * edit, and so on. Every plausible key is checked, because the point of
 * this function is to know whether the call touches a protected file.
 */
export function extractCallPath(input: any): string | undefined {
  if (!input || typeof input !== "object") return undefined
  const KEYS = ["path", "filePath", "file", "filename", "file_path", "target", "source", "dest"]
  for (const key of KEYS) {
    const value = input[key]
    if (typeof value === "string" && value) return value
  }
  return undefined
}

// ---------------------------------------------------------------------------
// snip — filter shell output before it ever becomes a tool result
// ---------------------------------------------------------------------------

/**
 * Make snip's rewritten command runnable by the shell that will run it.
 *
 * snip answers with the full command it wants executed, and it leads with the
 * path of its own binary: `"C:\\Users\\me\\go\\bin\\snip.exe" run -- git log`.
 * That answer is correct in the abstract and broken in practice, in two ways
 * that only show up on Windows:
 *
 *   - The path is quoted. A quoted path is not a command in PowerShell: it is
 *     a string expression, and the rest of the line is a parse error
 *     ("Unexpected token 'run'"). PowerShell needs the call operator to run a
 *     quoted path. cmd and bash do not.
 *   - The path comes back JSON-escaped, so it holds doubled backslashes.
 *     Windows tolerates those, but they are noise in every log line and in
 *     every error message the agent reads.
 *
 * So the path is dropped and replaced with one the target shell can run. The
 * bare name is the default, and it is the right default for a specific reason:
 * the plugin spawns snip by bare name, so the name is on the PATH the shell
 * inherits. No quoting, no escaping, and it works in cmd, PowerShell and bash
 * alike.
 *
 * The rest of snip's answer is kept verbatim, so a future snip that adds a
 * flag to `run` still works.
 *
 * `shell` is the shell binary the session uses, when it is known. It only
 * matters for an explicitly configured path containing spaces, where the
 * quoting has to match the shell.
 */
export function repairSnipCommand(rewritten: string, snipPath: string, shell?: string): string {
  // Split off snip's own path: quoted, or up to the first space.
  let rest: string
  if (rewritten.startsWith('"')) {
    const end = rewritten.indexOf('"', 1)
    if (end < 0) return rewritten
    rest = rewritten.slice(end + 1)
  } else {
    const space = rewritten.indexOf(" ")
    if (space < 0) return rewritten
    rest = rewritten.slice(space)
  }
  const bin = snipPath || "snip"
  if (!/\s/.test(bin)) return `${bin}${rest}`
  // An explicit path with spaces has to be quoted, and only PowerShell needs
  // the call operator to run a quoted path. cmd treats a leading `&` as a
  // command separator and would run an empty command first, so it must not
  // have one.
  const quoted = `"${bin.replace(/\\/g, "/")}"`
  const isPowerShell = shell === undefined || /powershell|pwsh/i.test(shell)
  return isPowerShell ? `& ${quoted}${rest}` : `${quoted}${rest}`
}

/**
 * Ask `snip` to rewrite a shell command so its output is filtered.
 *
 * snip speaks the Claude Code PreToolUse format, so the plugin can hand it a
 * bare command and get back either a rewritten one or nothing at all. Nothing
 * comes back for a command snip has no filter for, for a command whose output
 * feeds a pipe or a redirection, and for a command that would change the
 * meaning of the output. That is the behaviour we want: the rewrite is opt-in
 * per command, and an unfiltered command is still correct.
 *
 * The answer is repaired before it is returned, so what comes back is a
 * command the target shell can actually run.
 *
 * Returns undefined on any failure. A missing binary, a timeout, or a
 * malformed answer all mean the same thing: run the command as it was written.
 */
export function rewriteViaSnip(
  command: string,
  opts: { snipPath: string; snipTimeoutMs: number },
  shell?: string,
): Promise<string | undefined> {
  return new Promise((resolve) => {
    const bin = opts.snipPath || "snip"
    const child = spawn(bin, ["hook"], { timeout: opts.snipTimeoutMs, windowsHide: true })
    let out = ""
    let settled = false
    const finish = (value: string | undefined) => {
      if (settled) return
      settled = true
      resolve(value)
    }
    child.stdout?.on("data", (chunk: any) => {
      out += String(chunk)
    })
    child.on("error", () => finish(undefined))
    child.on("close", () => {
      if (!out.trim()) return finish(undefined)
      try {
        const parsed = JSON.parse(out)
        const rewritten = parsed?.hookSpecificOutput?.updatedInput?.command
        if (typeof rewritten !== "string" || rewritten === command) return finish(undefined)
        finish(repairSnipCommand(rewritten, opts.snipPath, shell))
      } catch {
        finish(undefined)
      }
    })
    child.stdin?.on("error", () => {})
    child.stdin?.end(JSON.stringify({ tool_name: "Bash", tool_input: { command } }))
  })
}

/**
 * Replace the text in a tool result with a digest, keeping the shape intact.
 *
 * A tool result can carry many text items. The digest replaces ALL of them,
 * and it goes into the FIRST slot — the rest are dropped. Writing the digest
 * into every slot would multiply it: a result split across 40 text items
 * would go from 80 KB to 40 times the digest, and the payload would GROW.
 *
 * Non-text items (a file reference, an image) are never touched. A file
 * entry is a reference, not a payload, and has to survive intact.
 *
 * Returns the new items and the number of bytes saved.
 */
export function elideContentItems(
  value: any[],
  digest: string,
): { value: any[]; saved: number } {
  const textItems = value.filter((item: any) => item?.type === "text")
  const original = textItems.reduce((n: number, item: any) => n + bytes(item.text ?? ""), 0)
  const out: any[] = []
  let placed = false
  for (const item of value) {
    if (item?.type !== "text") {
      out.push(item)
      continue
    }
    if (placed) continue
    placed = true
    out.push({ type: "text", text: digest })
  }
  return { value: out, saved: original - bytes(digest) }
}

/**
 * Write the full report to a file and return its path.
 *
 * The TUI renders a synthetic message's description in a single-line text
 * element, so a multi-line report loses everything below the first line. The
 * report therefore has to live somewhere that can hold all of it, and the
 * path goes in the one line the TUI can show.
 */
export function writeReportFile(directory: string, sessionID: string, text: string): string | undefined {
  try {
    fs.mkdirSync(directory, { recursive: true })
    const file = path.join(directory, `${sessionID}-report.txt`)
    fs.writeFileSync(file, text, "utf8")
    return file
  } catch {
    return undefined
  }
}

/**
 * The report's headline, on one line.
 *
 * This is what the TUI can actually show. It carries the two numbers that
 * matter — what was sent, and how much smaller it is than the untrimmed
 * history — and nothing else.
 */
export function reportSummary(text: string): string {
  const lines = text.split("\n")
  // The line under "This request (what was actually sent to the model):".
  const sent = lines.find((l) => /\(\d+% smaller than the untrimmed history\)/.test(l))
  const reasoning = lines.find((l) => /^\s+reasoning:/.test(l))
  const results = lines.find((l) => /^\s+tool results:/.test(l))
  const parts = [lines[0] ?? "Token slim"]
  if (sent) parts.push(sent.trim())
  if (reasoning) parts.push(reasoning.trim())
  if (results) parts.push(results.trim())
  return parts.join(" | ")
}

/**
 * Whether `snip` is usable on this machine.
 *
 * Probing the hook rather than the binary matters: a snip too old to have the
 * hook subcommand would otherwise leave every command unfiltered while looking
 * perfectly installed.
 */
export async function snipAvailable(opts: { snipPath: string; snipTimeoutMs: number }): Promise<boolean> {
  try {
    const rewritten = await rewriteViaSnip("git status", opts)
    return typeof rewritten === "string"
  } catch {
    return false
  }
}

// ---------------------------------------------------------------------------
// Behaviour brief — the cheapest saving, because it stops waste instead of
// compressing it.
// ---------------------------------------------------------------------------

export const BRIEF = [
  "## Context discipline",
  "",
  "Your context is re-sent on every turn, so each redundant byte is paid for repeatedly.",
  "",
  "- Read a file once. Reuse what you already read instead of re-reading it.",
  "- One larger read beats several small slices of the same file.",
  "- Search before reading: `grep` for a symbol, then read only the lines around it.",
  "- Shell output is filtered before you see it: noise is removed, errors and summaries are kept.",
  "  The result is not missing anything you need, and it is not a partial failure.",
  "- Do not re-run a command to recover output you already have.",
  "",
  "### Elided context",
  "",
  "Old parts of this conversation are shortened to save tokens. Every one of them says what it is:",
  "",
  "- `[slim: ... elided ... Full text: <path>]` — a result from earlier. The full text is in the file.",
  "  Read it if you need it; it is not missing anything, it is just not inlined.",
  "- A result with a `… N lines omitted …` marker kept its beginning and its end, and names the range",
  "  that was cut. Re-read that range with offset/limit if you need it.",
  "- Your own earlier reasoning is gone. That is deliberate: you have already acted on it.",
  "- Your own earlier prose is shortened to its beginning and its end, and says how much was cut.",
  "- An old tool call keeps its name and its shape, but a long argument in it is shortened. The",
  "  file it wrote is on disk, so read that if you need the exact text.",
  "",
  "None of this is a failure and none of it needs fixing. Use the files; do not go looking for a",
  "broken or missing section.",
].join("\n")

/**
 * A key for a piece of reasoning, used to count it once.
 *
 * The message id is the right key and is what the LLM frame always carries.
 * A message without one still has to be counted once, so the content is
 * fingerprinted instead: the first and last 64 characters and the length.
 * That is stable across requests, because the history does not change, and
 * it is cheap, because the whole text is not hashed on every request.
 */
export function reasoningKey(messageID: any, text: string): string {
  if (typeof messageID === "string" && messageID) return `m:${messageID}`
  const t = text ?? ""
  const head = t.slice(0, 64)
  const tail = t.length > 64 ? t.slice(-64) : ""
  return `t:${t.length}:${head}:${tail}`
}

// ---------------------------------------------------------------------------
// Per-session accounting
// ---------------------------------------------------------------------------

interface Stats {
  requests: number
  /** Payload of the last request before trimming, in bytes. */
  requestBefore: number
  /** Payload of the last request after trimming, in bytes. */
  requestAfter: number
  /** Largest untrimmed payload seen this session, in bytes. */
  peakBefore: number
  /** The trimmed payload of the request that had the largest untrimmed payload, in bytes. */
  peakAfter: number
  reasoningDropped: number
  reasoningBytesDropped: number
  resultsElided: number
  resultBytesDropped: number
  /** Results put back whole because they fell inside the kept region again. */
  resultsRestored: number
  /** Bytes those restores added back to the payload. */
  resultBytesRestored: number
  duplicatesCollapsed: number
  duplicateBytesDropped: number
  /** Tool calls whose bulky arguments were trimmed. */
  callsElided: number
  /** Bytes removed from tool-call arguments. */
  callBytesDropped: number
  /** Assistant text parts that were shortened. */
  textElided: number
  /** Bytes removed from assistant text. */
  textBytesDropped: number
  systemBytesDropped: number
  toolBytesDropped: number
  /** Bytes of the behaviour brief the plugin adds to every request. */
  briefBytes: number
  /** Bytes of shell output that reached the context after snip filtering. */
  snipBytesFiltered: number
  /** How many shell commands snip rewrote. */
  snipCommands: number
  /** Results left alone because their path or tool is protected. */
  protectedSkipped: number
  /** Bytes saved by tool, so the report can name the expensive one. */
  byTool: Record<string, number>
}

function emptyStats(): Stats {
  return {
    requests: 0,
    requestBefore: 0,
    requestAfter: 0,
    peakBefore: 0,
    peakAfter: 0,
    reasoningDropped: 0,
    reasoningBytesDropped: 0,
    resultsElided: 0,
    resultBytesDropped: 0,
    resultsRestored: 0,
    resultBytesRestored: 0,
    duplicatesCollapsed: 0,
    duplicateBytesDropped: 0,
    callsElided: 0,
    callBytesDropped: 0,
    textElided: 0,
    textBytesDropped: 0,
    systemBytesDropped: 0,
    toolBytesDropped: 0,
    briefBytes: 0,
    snipBytesFiltered: 0,
    snipCommands: 0,
    protectedSkipped: 0,
    byTool: {},
  }
}

/** Bytes kept from the start of a long string inside an old tool call. */
const CALL_ARG_KEEP = 200

/** A trim this small is not worth the marker that explains it. */
const CALL_TRIM_MIN = 128

/**
 * Trim the bulky string fields inside a tool call's arguments.
 *
 * The call itself is kept: the model needs to know it made the call, and the
 * provider needs the call to pair with its result. Only the arguments shrink,
 * and only the long strings inside them.
 *
 * A `write` carries the whole file in its arguments, so a session that writes
 * ten files replays all ten files on every turn forever. Trimming the
 * arguments keeps the shape — the same keys, the same types — so the call
 * still means what it meant.
 *
 * Whether a call is old enough to trim is decided by the caller, from the
 * byte budget. It used to be decided here as well, from the same budget, and
 * that double check was the bug: an old call of 7 KB sat under the 8 KB
 * budget on its own and was left whole, so two thousand sub-budget calls
 * replayed in full. A budget is a total, not a per-part ceiling.
 */
export function trimCallInput(input: any): { input: any; changed: boolean; saved: number } {
  if (!input || typeof input !== "object") return { input, changed: false, saved: 0 }
  let saved = 0
  const walk = (value: any, depth: number): any => {
    if (depth > 6) return value
    if (typeof value === "string") {
      if (bytes(value) <= CALL_ARG_KEEP) return value
      const kept = takeBytes(value, CALL_ARG_KEEP)
      // The marker is paid for out of the saving, so the number reported is
      // the number of bytes the payload actually lost.
      const next = `${kept}\n[slim: ${fmt(bytes(value) - bytes(kept))} of this argument elided]`
      saved += bytes(value) - bytes(next)
      return next
    }
    if (Array.isArray(value)) return value.map((v) => walk(v, depth + 1))
    if (value && typeof value === "object") {
      const out: any = {}
      for (const [k, v] of Object.entries(value)) out[k] = walk(v, depth + 1)
      return out
    }
    return value
  }
  const next = walk(input, 0)
  if (saved < CALL_TRIM_MIN) return { input, changed: false, saved: 0 }
  return { input: next, changed: true, saved }
}

/**
 * Decide what to keep, oldest-first, under a byte budget.
 *
 * This is the core of the plugin and it works on BYTES, not on message
 * counts. A session's content is wildly uneven: the median reasoning part is
 * 1.5 KB and the largest is 70 KB, and a tool result can be 200 bytes or
 * 200 KB. Any rule expressed in message counts therefore keeps a wildly
 * different amount from one session to the next, and the saving swings from
 * turn to turn.
 *
 * The rule is simple and it is the same for every kind of content:
 *
 *   - Walk the history from the NEWEST end backwards.
 *   - Keep everything until the budget for that kind of content is spent.
 *   - Everything older than that is elided.
 *
 * Two floors sit under it:
 *
 *   - The "keep" budget is a hard floor for the most recent content. The
 *     reasoning the model is mid-way through is kept whatever else happens,
 *     because cutting it is what makes a model repeat itself.
 *   - A part carrying a provider signature is never touched, because it has
 *     to be replayed byte-identically.
 *
 * The order of the walk is what makes the result readable: the newest content
 * is the first thing kept, so what survives is the context the model is
 * actually working in, and what is dropped is the history it has already
 * acted on.
 */
export interface KeepPlan {
  /** Index into the flattened part list; parts before it are elided. */
  cut: number
}

/**
 * Flatten a message list into the parts the budgets apply to, oldest first.
 *
 * The parts are returned with the message they belong to, so a caller can put
 * a part back where it came from.
 *
 * The index is the part's position in the WHOLE history, not in its message.
 * The budget pass walks the messages and counts parts in the same order, so
 * the two have to agree; a per-message index would let every message claim
 * the same slot.
 */
export function flattenParts(messages: any[]): { message: any; part: any; index: number }[] {
  const out: { message: any; part: any; index: number }[] = []
  for (const message of messages) {
    const content = message?.content
    if (!Array.isArray(content)) continue
    for (const part of content) {
      out.push({ message, part, index: out.length })
    }
  }
  return out
}

/**
 * The byte cost of a part, as it counts against a budget.
 */
export function partCost(part: any): number {
  if (!part) return 0
  if (part.type === "text" || part.type === "reasoning") return bytes(part.text ?? "")
  if (part.type === "tool-result") return bytes(JSON.stringify(part.result ?? {}))
  if (part.type === "tool-call") return bytes(JSON.stringify(part.input ?? {}))
  return bytes(JSON.stringify(part))
}

/**
 * Round a byte position down to a multiple of the step.
 *
 * This is what makes the saving CONSISTENT without busting the prompt cache.
 *
 * The cut is a position in the byte stream, measured from the START of the
 * history. The history only ever grows at the end, so a position measured from
 * the start means the same thing on every request. Rounding that position
 * down to a multiple of the step means the cut only MOVES when a whole step
 * has accumulated, and between moves the request is byte-identical and the
 * provider's cache still hits.
 *
 * A sliding window would move the cut on every single turn, so every turn
 * would be a full-price re-read of the whole history. That is more expensive
 * than not trimming at all.
 */
export function quantize(position: number, step: number): number {
  if (step <= 0) return position
  return Math.max(0, Math.floor(position / step) * step)
}

/**
 * What to do with one part of one kind of content.
 */
export type Band = "keep" | "digest" | "elide"

/**
 * Decide each part's fate, from a byte budget and a stable cut.
 *
 * The parts of one kind are laid out in a byte stream, oldest first. The
 * budget says how much of the NEWEST end is kept whole, the next budget
 * says how much of the middle gets a digest, and the rest is old: a pointer,
 * a drop, a trim.
 *
 * The boundaries between the bands are byte positions measured from the
 * START of the stream, and they are quantized to a step. The history only
 * grows at the end, so a position measured from the start means the same
 * thing on every request. Rounding it down to a step means the boundaries
 * only MOVE when a whole step has accumulated, and between moves every part
 * keeps the same fate and the request is byte-identical — so the provider's
 * prompt cache still hits.
 *
 * A part is placed by where it STARTS. A part that straddles a boundary
 * belongs to the older band, so one huge part cannot smuggle itself into the
 * kept region whole. The kept region is the newest `budget` bytes, give or
 * take one part and one step.
 *
 * Two things are kept whatever the budget says: the newest part of each
 * kind, because it is what the model is looking at right now, and — for
 * reasoning — the newest `reasoningKeepBytes`, which is the model's current
 * train of thought.
 */
export function assignBands(
  parts: readonly { part: any; index: number }[],
  opts: SlimOptions,
  kind: "reasoning" | "tool-result" | "tool-call" | "text",
): Map<number, Band> {
  const bands = new Map<number, Band>()
  if (parts.length === 0) return bands
  const budget = budgetForKind(kind, opts)
  const step = stairForKind(kind, opts)
  const digestBudget = kind === "tool-result" ? opts.resultDigestBytes : 0

  // The stream: each part's start position, measured from the start.
  const starts: number[] = []
  const costs: number[] = []
  let total = 0
  for (const p of parts) {
    starts.push(total)
    const cost = partCost(p.part)
    costs.push(cost)
    total += cost
  }

  // The boundaries, quantized so they only move on a step boundary. Rounding
  // down keeps a little MORE than the budget, which is the safe direction:
  // the cut can never leave the model with less than it asked for.
  const keepBoundary = quantize(total - budget, step)
  const digestBoundary = Math.max(0, quantize(total - budget - digestBudget, step))

  for (let i = 0; i < parts.length; i++) {
    const start = starts[i]
    const band: Band = start >= keepBoundary ? "keep" : start >= digestBoundary ? "digest" : "elide"
    bands.set(parts[i].index, band)
  }

  // The newest part is kept whole whatever the budget says. It is the one
  // the model is looking at right now, and it is usually the biggest thing
  // in the history, so the budget alone would drop it.
  bands.set(parts[parts.length - 1].index, "keep")

  // The floor: the newest content is kept whole whatever the budget says.
  // For reasoning this is the model's current train of thought, and it is not
  // negotiable. It applies to the newest parts only, so it cannot be used to
  // keep the whole history.
  const floor = floorForKind(kind, opts)
  if (floor > 0) {
    let spent = 0
    for (let i = parts.length - 1; i >= 0; i--) {
      if (i < parts.length - 1 && spent + costs[i] > floor) break
      spent += costs[i]
      bands.set(parts[i].index, "keep")
    }
  }
  return bands
}

/**
 * The floor for a kind of content: the newest content is kept whatever the
 * budget says.
 */
export function floorForKind(kind: string, opts: SlimOptions): number {
  if (kind === "reasoning") return opts.reasoningKeepBytes
  return 0
}

/**
 * The byte budget for a kind of content: how much of it is kept whole.
 *
 * The reasoning budget is never below the keep floor. The floor is the
 * model's current train of thought, and it is not negotiable: if the budget
 * were smaller than the floor, the floor wins.
 */
export function budgetForKind(kind: string, opts: SlimOptions): number {
  if (kind === "reasoning") return Math.max(opts.reasoningBytes, opts.reasoningKeepBytes)
  if (kind === "tool-result") return opts.resultKeepBytes
  if (kind === "tool-call") return opts.callKeepBytes
  if (kind === "text") return opts.textKeepBytes
  return 0
}

/**
 * The step for a kind of content: how much has to accumulate before the cut
 * moves.
 */
export function stairForKind(kind: string, opts: SlimOptions): number {
  return opts.stairBytes
}

export class Ledger {
  private stats = emptyStats()
  private seen = new Map<string, Set<string>>()
  /** Ids of the parts already counted as elided, so each one counts once. */
  private elided = new Set<string>()
  /** Ids of the parts already counted as restored, so each one counts once. */
  private restored = new Set<string>()

  /**
   * Claim a history tool result for the elided counter. Returns true the
   * first time and false afterwards.
   *
   * The same result is re-elided on every request, because the `context`
   * hook hands over a fresh, untrimmed copy of the history each time. Without
   * this, one 50 KB result would be counted as 50 KB saved on every request
   * and the report would inflate the saving by the number of turns.
   */
  claimElided(key: string) {
    if (this.elided.has(key)) return false
    this.elided.add(key)
    return true
  }

  /** Claim a part for the restored counter. Same once-only rule. */
  claimRestored(key: string) {
    if (this.restored.has(key)) return false
    this.restored.add(key)
    return true
  }

  /** Bucket of content hashes already seen for a tool in this session. */
  seenFor(tool: string) {
    let bucket = this.seen.get(tool)
    if (!bucket) {
      bucket = new Set<string>()
      this.seen.set(tool, bucket)
    }
    return bucket
  }

  bump(patch: Partial<Stats>) {
    for (const [k, v] of Object.entries(patch)) {
      // byTool is a map, not a number, and adding to it here would corrupt it.
      if (k === "byTool" || typeof v !== "number") continue
      ;(this.stats as unknown as Record<string, number>)[k] += v
    }
  }

  /**
   * Record bytes saved by a named tool.
   *
   * The report names the expensive tool, because "you saved 40%" is not
   * actionable on its own. "Your `bash` results cost 30 KB and your `read`
   * results cost 2 KB" is.
   */
  bumpTool(tool: string, bytesSaved: number) {
    if (!tool || bytesSaved <= 0) return
    const current = this.stats.byTool
    current[tool] = (current[tool] ?? 0) + bytesSaved
  }

  set(patch: Partial<Stats>) {
    Object.assign(this.stats, patch)
  }

  /** Merge previously persisted numbers back in, after a restart. */
  restore(previous: Partial<Stats>) {
    for (const [k, v] of Object.entries(previous)) {
      if (k === "byTool") {
        const byTool = (v as Record<string, number>) ?? {}
        for (const [tool, saved] of Object.entries(byTool)) {
          this.stats.byTool[tool] = (this.stats.byTool[tool] ?? 0) + saved
        }
        continue
      }
      if (typeof v === "number") (this.stats as unknown as Record<string, number>)[k] += v
    }
  }

  get() {
    return this.stats
  }

  /**
   * Forget the markers for messages that left the history.
   *
   * The `context` hook gets a fresh copy of the history every request, so
   * the decision to elide is recomputed from scratch each time. The markers
   * are here so a piece of content is COUNTED once, not once per request.
   */
  retain(ids: ReadonlySet<string>) {
    for (const set of [this.elided, this.restored]) {
      for (const key of set) {
        if (!key.startsWith("m:")) continue
        if (!ids.has(key.slice(2))) set.delete(key)
      }
    }
  }
}

// ---------------------------------------------------------------------------
// Persisted statistics
// ---------------------------------------------------------------------------

/**
 * Read the persisted counters for a session.
 *
 * A restart loses the in-memory ledger, and with it the running totals. The
 * previous session's numbers are restored so the report keeps telling the
 * truth across a restart.
 *
 * Any failure to read is not an error: a missing file, a corrupt file, and an
 * unwritable directory all mean "start from zero".
 */
export function readPersistedStats(directory: string, sessionID: string): Partial<Stats> | undefined {
  try {
    const file = path.join(directory, `${sessionID}.json`)
    if (!fs.existsSync(file)) return undefined
    const parsed = JSON.parse(fs.readFileSync(file, "utf8"))
    if (!parsed || typeof parsed !== "object") return undefined
    return parsed as Partial<Stats>
  } catch {
    return undefined
  }
}

/**
 * Write the counters for a session.
 *
 * Written on a timer rather than on every request: a session makes many
 * requests and the numbers only need to survive a crash, not be exact to the
 * byte.
 */
export function writePersistedStats(directory: string, sessionID: string, stats: Partial<Stats>) {
  try {
    fs.mkdirSync(directory, { recursive: true })
    const file = path.join(directory, `${sessionID}.json`)
    // Write to a sibling and rename, so a crash mid-write cannot leave a
    // half-written file that fails to parse on the next start.
    const tmp = `${file}.tmp`
    fs.writeFileSync(tmp, JSON.stringify(stats), "utf8")
    fs.renameSync(tmp, file)
  } catch {
    // A stats file that cannot be written is not worth failing a session over.
  }
}

// ---------------------------------------------------------------------------
// Tool result digesting
// ---------------------------------------------------------------------------

/**
 * Write the full text next to the digest, and return its path.
 *
 * The name is content-addressed. The same result is re-digested on every
 * request — the `context` hook hands over a fresh copy of the history each
 * time — so a timestamped name wrote a new file per request. A long session
 * produced thousands of near-identical copies of the same output: 1846 files
 * and 35 MB for one session, most of them duplicates.
 *
 * Addressing by content fixes that, and has a second benefit: the digest text
 * is stable across requests, because the path it names is stable. A stable
 * digest keeps the cached prefix intact.
 */
export function writeFullOutput(directory: string, name: string, text: string) {
  try {
    fs.mkdirSync(directory, { recursive: true })
    const safe = name.replace(/[^A-Za-z0-9_.-]/g, "_").slice(0, 24)
    // Address the file by its content, and by nothing else. The session is not
    // in the name for two reasons: the same output read in two sessions is one
    // file, and the name is replayed in the digest on every request, so every
    // byte of it is a byte of context.
    const hash = createHash("sha1").update(text).digest("hex").slice(0, 12)
    const file = path.join(directory, `${hash}_${safe}.txt`)
    // Same content, same path: the file is already there. Rewriting it would
    // cost the same I/O for no gain.
    if (!fs.existsSync(file)) fs.writeFileSync(file, text, "utf8")
    return file
  } catch {
    return undefined
  }
}

/**
 * Keep the full-output directory under a byte cap, dropping the oldest files.
 *
 * The directory is a cache of what the digest points at, not an archive. Once
 * the digest has been read the full copy is dead weight, and without a cap the
 * directory grows for the lifetime of the install.
 *
 * The scan is throttled to once a minute. It used to run for EVERY digested
 * result, and a digested result is re-digested on every request: a session
 * with a thousand results and a few thousand files spent the whole request in
 * readdir and stat. The cap is 256 MiB by default; a minute of drift costs
 * nothing.
 */
let lastDiskScan = 0
const DISK_SCAN_MS = 60_000

export function enforceDiskCap(directory: string, maxBytes: number, keep?: Set<string>) {
  const now = Date.now()
  if (now - lastDiskScan < DISK_SCAN_MS) return
  lastDiskScan = now
  let entries: fs.Dirent[]
  try {
    entries = fs.readdirSync(directory, { withFileTypes: true })
  } catch {
    return
  }
  let files: { path: string; size: number; mtime: number }[] = []
  let total = 0
  for (const entry of entries) {
    if (!entry.isFile()) continue
    const file = path.join(directory, entry.name)
    try {
      const stat = fs.statSync(file)
      files.push({ path: file, size: stat.size, mtime: stat.mtimeMs })
      total += stat.size
    } catch {
      // A file that vanished between the listing and the stat is not a problem.
    }
  }
  if (total <= maxBytes) return
  // Oldest first, but a file the request still points at goes last: deleting
  // it turns a digest into a dead end, and a dead end is data loss. The cap
  // is still hard — if every file is referenced, the oldest go anyway.
  const held = (f: { path: string }) => (keep?.has(f.path) ? 1 : 0)
  files.sort((a, b) => held(a) - held(b) || a.mtime - b.mtime)
  for (const file of files) {
    if (total <= maxBytes) break
    try {
      fs.unlinkSync(file.path)
      total -= file.size
    } catch {
      // Already gone.
    }
  }
}

/**
 * Replace old content with a one-line pointer to the full text on disk.
 *
 * A digest of a result keeps its head and its tail, and that is worth doing
 * for a result the model is about to read. It is not worth doing for a result
 * from fifty turns ago: the model has already acted on it, and the digest
 * costs as much as the original.
 *
 * The numbers make this unavoidable. A session with 630 tool results spends
 * 1652 KB on them. Digesting all of them at a 4.6 KB head+tail costs 863 KB
 * — half the original, for content the model already used. A pointer costs
 * about 150 bytes, so all 630 cost 94 KB.
 *
 * So the rule is: the newest results are digested, the old ones are pointed
 * at. The model can still read the file, so nothing is lost — it just has to
 * ask for it.
 *
 * The pointer names the file, so the model can re-read it. A pointer to
 * nothing would be data loss.
 */
export function elideToPointer(
  text: string,
  opts: SlimOptions,
  toolName: string,
  size: number,
): string | undefined {
  let pointer = ""
  const existing = markerPath(text)
  if (existing) {
    // The content is already a digest that points at the full text.
    pointer = ` Full text: ${existing}`
  } else if (opts.keepFullOutput) {
    const file = writeFullOutput(opts.directory, toolName, text)
    if (file) pointer = ` Full text: ${file}`
  }
  // A pointer without a file is a pointer to nothing: the content is gone and
  // the model is told it is gone. That is worse than a digest, which at
  // least keeps the head and the tail, so the caller falls back to one.
  if (!pointer) return undefined
  const kb = (size / 1024).toFixed(1)
  return `[slim: ${toolName} result, ${kb} KB, elided - read earlier and acted on.${pointer}]`
}

/**
 * The file a slim marker points at, if the text carries one.
 *
 * Both shapes are recognised: a head+tail digest (`Full output:`) and a bare
 * pointer (`Full text:`). The path stops at whitespace and at the closing
 * bracket, so a pointer to a pointer cannot swallow the bracket and grow one
 * more `]` on every request.
 */
export function markerPath(text: string): string | undefined {
  return /\[slim:[^\]]*?Full (?:output|text): ([^\s\]]+)/.exec(text)?.[1]
}

/**
 * Read back the full text a marker points at, if it is cheap enough to send.
 *
 * `context` hands over a fresh copy of the history on every request, and the
 * elision is not persisted — except at source, where the trimmed text is what
 * the session stores. Undo rewinds the session: a result that was elided when
 * it was old can be the newest thing the model has again, and the database
 * still holds the marker. Restoring it from disk is the only way the model
 * gets its working context back.
 *
 * A restore is a repair, not a way around the budget: the file has to be in
 * the plugin's own cache directory and no bigger than the kept region. Anything
 * larger stays a digest — the model can still read it in slices.
 */
export function readRestorable(text: string, opts: SlimOptions): string | undefined {
  if (!opts.keepFullOutput) return undefined
  const file = markerPath(text)
  if (!file) return undefined
  try {
    const root = path.resolve(opts.directory)
    const full = path.resolve(file)
    if (!full.startsWith(root + path.sep)) return undefined
    const stat = fs.statSync(full)
    if (!stat.isFile() || stat.size > opts.resultKeepBytes) return undefined
    return fs.readFileSync(full, "utf8")
  } catch {
    return undefined
  }
}

/** Which markers in the kept region can be made whole again, and with what. */
export function planRestores(
  flat: readonly { part: any }[],
  bands: Map<any, Band>,
  opts: SlimOptions,
): Map<any, string> {
  const plan = new Map<any, string>()
  if (!opts.keepFullOutput) return plan
  // The ceiling is the kept region itself: undo may bring a few results back,
  // but never a second copy of the whole history. Newest first, so the part
  // the model is looking at wins the budget.
  let left = opts.resultKeepBytes
  for (let i = flat.length - 1; i >= 0 && left > 0; i--) {
    const part = flat[i].part
    if (part?.type !== "tool-result") continue
    // The kept region is a suffix of the stream, so the first result outside
    // it ends the walk. Reading a marker far back in the history would be a
    // file read per result per request for nothing.
    if (bands.get(part) !== "keep") break
    const text = resultText(part)
    if (!text) continue
    const full = readRestorable(text, opts)
    if (full === undefined) continue
    const size = bytes(full)
    if (size > left) continue
    plan.set(part, full)
    left -= size
  }
  return plan
}

/** Replace a result part's text, keeping the part's shape and its file items. */
export function restorePart(part: any, full: string): any {
  const result = part.result
  if (result.type === "text") return { ...part, result: { ...result, value: full } }
  if (!Array.isArray(result.value)) return part
  let done = false
  const value = result.value.map((item: any) => {
    if (done || item?.type !== "text") return item
    done = true
    return { ...item, text: full }
  })
  return { ...part, result: { ...result, value } }
}

/** The text of a tool result, whether it is a plain string or a content list. */
export function resultText(part: any): string | undefined {
  const result = part?.result
  if (result?.type === "text" && typeof result.value === "string") return result.value
  if (result?.type === "content" && Array.isArray(result.value))
    return result.value
      .filter((item: any) => item?.type === "text")
      .map((item: any) => item.text)
      .join("\n")
  return undefined
}

/**
 * Replace old image items with a one-line pointer to the file they came from.
 *
 * A `read` of a png returns the bytes inline, as `{type:"file", uri:"data:...;base64,..."}`
 * — about 170 KB of base64 per screenshot. None of the rules above touch it:
 * they digest *text*, and this result has none. One measured session carried
 * 280 of them, 47 MB of a 48 MB payload, and the plugin cut 2% of that session.
 *
 * No copy is written to the cache directory. The item already names the file it
 * was read from, and that file is where the bytes came from, so the pointer
 * says so and the model can read it again. Writing 47 MB into the cache to be
 * able to re-read what is already on disk would be the plugin's own bloat.
 *
 * Images are not restored by undo either: the file still exists, so a re-read
 * is the recovery path, and the newest images are never touched — the result
 * bands keep them.
 */
export function elideImages(part: any, toolName: string) {
  const result = part?.result
  if (result?.type !== "content" || !Array.isArray(result.value)) return undefined
  let saved = 0
  let touched = false
  const created: any[] = []
  const value = result.value.map((item: any) => {
    if (item?.type !== "file") return item
    const mime = String(item.mime ?? item.mimeType ?? "")
    if (!mime.startsWith("image/")) return item
    // An inline image is the only shape worth replacing: one with a plain URL
    // is already a reference and costs nothing to keep.
    const inline = typeof item.data === "string" || typeof item.uri === "string" || typeof item.url === "string"
    if (!inline) return item
    const kb = (bytes(JSON.stringify(item)) / 1024).toFixed(0)
    const name = typeof item.name === "string" && item.name.length > 0 ? item.name : undefined
    const where = name ? ` from ${name}` : ""
    // No "Full text:" here on purpose: `markerPath` would take it for a text
    // digest and try to restore a png as utf8.
    const pointer = {
      type: "text",
      text: `[slim: ${toolName} image, ${mime}, ${kb} KB, elided - read${where} again if needed]`,
    }
    saved += bytes(JSON.stringify(item)) - bytes(JSON.stringify(pointer))
    touched = true
    created.push(pointer)
    return pointer
  })
  if (!touched) return undefined
  return { part: { ...part, result: { ...result, value } }, saved, created }
}

/**
 * The head and tail budgets for a result of a given size.
 *
 * A digest has to be SMALLER than the result it replaces, or the plugin is
 * making the payload bigger. A fixed head+tail of 4.5 KB is fine for a 50 KB
 * result and absurd for a 4 KB one — and a session with hundreds of small
 * results ends up paying 4.5 KB per digest, which is more than the results
 * themselves.
 *
 * So the digest scales: a big result gets a generous excerpt, a small one
 * gets a small excerpt, and the excerpt is never more than a third of the
 * original. The model still gets the beginning and the end — the two places
 * the answer usually is — and the payload always shrinks.
 */
export function digestBudgets(size: number, opts: SlimOptions): { head: number; tail: number } {
  // A digest of a result this small is not worth making.
  const cap = Math.max(0, Math.floor(size / 3))
  return {
    head: Math.min(opts.headBytes, cap),
    tail: Math.min(opts.tailBytes, Math.max(0, Math.floor(cap / 2))),
  }
}

/**
 * Replace an oversized tool result with a digest plus a pointer to the full text.
 *
 * The digest keeps the head and the tail as *contiguous line ranges* and says
 * exactly which lines are missing. Splicing the head and the tail together
 * without that context makes the result look like it has a broken or corrupt
 * region, and a model that sees a file that jumps from line 10 to line 718
 * goes looking for the anomaly instead of using the data it has.
 */
export function digestResult(text: string, opts: SlimOptions, toolName: string) {
  const total = bytes(text)
  const budgets = digestBudgets(total, opts)
  const lines = text.split("\n")
  const lineCount = lines.length

  // Head: whole lines within the budget. The FIRST line is always taken -
  // even when it alone busts the budget - because a digest that starts
  // mid-file leaves the model with no idea what it is looking at. A line
  // too big for the budget is clipped to a byte window, never dropped:
  // losing a whole line is worse than keeping a truncated one.
  const headParts: string[] = []
  let used = 0
  let clippedHead = false
  for (let i = 0; i < lines.length; i++) {
    const size = bytes(lines[i]) + 1
    if (headParts.length > 0 && used + size > budgets.head) break
    if (headParts.length === 0 && size > budgets.head) {
      headParts.push(clipBytes(lines[i], budgets.head))
      clippedHead = true
      break
    }
    headParts.push(lines[i])
    used += size
  }
  const headEnd = headParts.length

  // Tail: the mirror of the head. The last line is always taken, and the
  // tail never overlaps the head.
  const tailParts: string[] = []
  let tailStart = lines.length
  used = 0
  let clippedTail = false
  for (let i = lines.length - 1; i >= headEnd; i--) {
    const size = bytes(lines[i]) + 1
    if (tailParts.length > 0 && used + size > budgets.tail) break
    if (tailParts.length === 0 && size > budgets.tail) {
      tailParts.unshift(clipBytes(lines[i], budgets.tail))
      clippedTail = true
      tailStart = i
      break
    }
    tailParts.unshift(lines[i])
    used += size
    tailStart = i
  }

  const head = headParts.join("\n")
  const tail = tailParts.join("\n")
  const omitted = tailStart - headEnd

  let pointer = ""
  if (opts.keepFullOutput) {
    const file = writeFullOutput(opts.directory, toolName, text)
    if (file) pointer = ` Full output: ${file}`
  }

  let clipNote = ""
  if (clippedHead || clippedTail) {
    const which = clippedHead && clippedTail ? "First and last line" : clippedHead ? "First line" : "Last line"
    clipNote = ` ${which} truncated to fit the budget.`
  }

  // State the omission in the same shape as the content around it, so the
  // model can tell what it is looking at.
  const marker =
    omitted > 0
      ? `[slim: ${total} bytes total, ${lineCount} lines. Lines ${headEnd + 1}-${tailStart} (${omitted} lines) omitted to save context.${clipNote}${pointer} Re-read with offset/limit for that range, or read the file.]`
      : `[slim: ${total} bytes / ${lineCount} lines.${clipNote}${pointer} Re-run with narrower output, or read the file.]`

  // The tail is included whenever it is non-empty, even when nothing was
  // omitted: a short result can have its last line fall outside the head.
  const digest = tail ? `${head}\n${marker}\n${tail}` : `${head}\n${marker}`

  // The marker costs ~150 bytes. On a result barely over the threshold the
  // digest can come out larger than the original, which would be a
  // regression, not a saving.
  return bytes(digest) < total ? digest : text
}

/**
 * Digest a JSON tool result while keeping it valid JSON.
 *
 * Many tools return JSON. A line-based digest of a JSON document produces a
 * broken one: the head is `{`, the tail is `}`, and the model gets something
 * that looks like malformed data rather than a summary of it. Models that are
 * told to parse JSON will fail on it, and a failed parse costs a whole turn.
 *
 * So for JSON the digest keeps the structure and shortens the leaves. The
 * result is still valid JSON, still says what the tool said, and names the
 * keys whose values were shortened.
 */
export function digestJson(text: string, opts: SlimOptions, toolName: string) {
  let parsed: any
  try {
    parsed = JSON.parse(text)
  } catch {
    return undefined
  }
  // Only objects and arrays are worth walking. A bare scalar is short enough
  // to have passed the size check already.
  if (parsed === null || typeof parsed !== "object") return undefined

  const truncated: string[] = []
  const shrink = (value: any, depth: number): any => {
    if (depth > 6) return "..."
    if (typeof value === "string") {
      if (bytes(value) <= 400) return value
      truncated.push(value.slice(0, 40))
      return `${takeBytes(value, 400)} [slim: shortened]`
    }
    if (Array.isArray(value)) {
      // Keep a bounded prefix of a long array, and say how long it was.
      const head = value.slice(0, 20).map((v) => shrink(v, depth + 1))
      if (value.length > 20) {
        truncated.push(`array of ${value.length}`)
        head.push(`[slim: ${value.length - 20} more items]`)
      }
      return head
    }
    if (value && typeof value === "object") {
      const out: any = {}
      let n = 0
      for (const [k, v] of Object.entries(value)) {
        if (n >= 40) {
          truncated.push(`object with ${Object.keys(value).length} keys`)
          out["[slim: more keys]"] = "..."
          break
        }
        out[k] = shrink(v, depth + 1)
        n++
      }
      return out
    }
    return value
  }

  const shaped = shrink(parsed, 0)
  let json = JSON.stringify(shaped, null, 2)
  if (bytes(json) > opts.resultBytes) {
    // Still too big: drop the pretty-printing before touching the content.
    json = JSON.stringify(shaped)
  }
  if (bytes(json) >= bytes(text)) return undefined

  let pointer = ""
  if (opts.keepFullOutput) {
    const file = writeFullOutput(opts.directory, toolName, text)
    if (file) pointer = ` Full output: ${file}`
  }

  const note = truncated.length
    ? ` [slim: ${truncated.length} value(s) shortened; structure preserved]`
    : ""
  const digest = `${json}${note}${pointer}`
  return bytes(digest) < bytes(text) ? digest : undefined
}

// ---------------------------------------------------------------------------
// Plugin
// ---------------------------------------------------------------------------

export default {
  id: "opencode-token-slim",

  async setup(ctx: any) {
    const opts = resolveOptions(ctx.options)
    if (!opts.enabled) return

    const ledgers = new Map<string, Ledger>()
    const ledgerFor = (sessionID: string) => {
      let l = ledgers.get(sessionID)
      if (!l) {
        l = new Ledger()
        // A restart loses the in-memory ledger, and with it the running totals.
        // The previous session's numbers are restored so the report keeps
        // telling the truth across a restart.
        const restored = readPersistedStats(opts.statsDirectory, sessionID)
        if (restored) l.restore(restored)
        ledgers.set(sessionID, l)
      }
      return l
    }

    // Persist the counters on a timer, so a crash does not lose the session's
    // numbers and a restart can restore them.
    const persistTimer = setInterval(() => {
      for (const [sessionID, ledger] of ledgers) {
        writePersistedStats(opts.statsDirectory, sessionID, ledger.get())
      }
    }, 30_000)
    persistTimer.unref?.()

    // -- 0. snip: filter shell output before it ever becomes a tool result.
    //
    //     This runs BEFORE the tool executes, so the filtered output is what
    //     the tool returns and what gets persisted. The alternative — digesting
    //     the result afterwards — can only cut it, while snip can replace it
    //     with something more informative than the original: a `go test` that
    //     printed 18 package lines becomes a count of 1125 passing tests.
    //
    //     It is the only optimisation here that cannot cost the model anything,
    //     because it never touches the model's context or its instructions.
    //
    //     The shell the session uses is learned from the shell hook. It is only
    //     needed to quote an explicitly configured snip path that contains
    //     spaces, but it is cheap to learn and it is the difference between a
    //     command that runs and a command that fails to parse.
    let shell: string | undefined
    if (ctx.shell?.hook) {
      await ctx.shell.hook("create.before", (event: any) => {
        if (typeof event?.shell === "string" && event.shell) shell = event.shell
      })
    }
    if (opts.snip) {
      const available = await snipAvailable({ snipPath: opts.snipPath, snipTimeoutMs: opts.snipTimeoutMs })
      if (available) {
        await ctx.tool.hook("execute.before", async (event: any) => {
          if (event?.tool !== "shell") return
          const input = event.input as { command?: unknown } | undefined
          const command = input?.command
          if (!command || typeof command !== "string") return
          const rewritten = await rewriteViaSnip(
            command,
            {
              snipPath: opts.snipPath,
              snipTimeoutMs: opts.snipTimeoutMs,
            },
            shell,
          )
          if (rewritten) event.input = { ...input, command: rewritten }
        })
      } else {
        console.warn("[slim] snip hook unavailable (binary missing or too old) — shell filtering disabled")
      }
    }

    // -- 1. Tool results: trim at source so the trimmed text is what persists.
    //
    //     One handler does both jobs. Registering `execute.after` twice would
    //     work on a runtime that keeps every handler for a hook, but it makes
    //     the two halves order-dependent: the second one has to see the
    //     first one's output. One handler has no such dependency, and it
    //     cannot be silently dropped by a host that keeps only the last
    //     registration.
    await ctx.tool.hook("execute.after", (event: any) => {
      if (event.status !== "completed") return
      const content = event.result?.content
      if (!Array.isArray(content)) return

      const ledger = ledgerFor(event.sessionID)
      let next: any[] = content
      let changed = false

      // A protected path or tool is never trimmed. The plugin's own source is
      // protected by default, so an agent editing the plugin always reads the
      // real file.
      if (opts.protectTools.includes(event.tool)) {
        // Nothing to do, but the shell accounting below still needs to run.
      } else {
        // The whole result is measured as one piece. A result split across
        // many text items is ONE result, and the budget applies to it as a
        // whole: measuring each item on its own would let forty 2 KB items
        // through a 12 KB budget untouched.
        const textItems = content.filter((item: any) => item?.type === "text")
        const text = textItems.map((item: any) => item.text ?? "").join("\n")
        const size = bytes(text)

        if (text && size > opts.resultBytes) {
          const sourcePath = extractPath(text)
          if (sourcePath && isProtectedPath(sourcePath, opts.protectPaths)) {
            ledger.bump({ protectedSkipped: 1 })
          } else {
            // Same tool, same content, already seen this session.
            const key = createHash("sha1").update(text).digest("hex")
            const bucket = ledger.seenFor(event.tool)
            if (size >= opts.duplicateBytes && bucket.has(key)) {
              const trimmed = `[slim: duplicate ${event.tool} result (${size} bytes, identical to an earlier result in this session).]`
              ledger.bump({ duplicatesCollapsed: 1, duplicateBytesDropped: size - bytes(trimmed) })
              ledger.bumpTool(event.tool, size - bytes(trimmed))
              const elided = elideContentItems(content, trimmed)
              next = elided.value
              changed = true
            } else {
              if (size >= opts.duplicateBytes && bucket.size < 256) bucket.add(key)
              const digest =
                digestJson(text, opts, event.tool) ?? digestResult(text, opts, event.tool)
              const elided = elideContentItems(content, digest)
              const saved = elided.saved
              ledger.bump({ resultsElided: 1, resultBytesDropped: saved })
              ledger.bumpTool(event.tool, saved)
              next = elided.value
              changed = true
            }
          }
        }
      }

      if (changed) event.result.content = next

      // -- 1b. snip's effect on the shell output.
      //
      //     The plugin rewrites the command before it runs, so the result that
      //     arrives is already filtered and the unfiltered size is gone. The
      //     bytes that were removed are therefore not measurable from here, and
      //     are not guessed at: snip keeps its own tracking database and
      //     reports the real number through `snip gain`.
      //
      //     What is measured here is what the plugin can honestly measure: how
      //     many commands were filtered, and how many bytes of shell output
      //     actually reached the context as a result. The second number is a
      //     cost, not a saving — it is the price of the filtered output, and
      //     the report shows it next to the saving so the two can be compared.
      if (!opts.snip || event.tool !== "shell") return
      const total = next.reduce(
        (n: number, item: any) => n + (item?.type === "text" ? bytes(item.text ?? "") : 0),
        0,
      )
      if (total <= 0) return
      ledger.bump({ snipCommands: 1, snipBytesFiltered: total })
    })

    // -- 2. Request-level diet: reasoning, system prompt, tool schemas.
    await ctx.session.hook("context", (event: any) => {
      const ledger = ledgerFor(event.sessionID)
      const messages = event.messages ?? []

      // 2z. Provider item ids.
      //
      //     The OpenAI Responses API replays the provider's own item ids with
      //     the history. Those ids are addresses in the resource that minted
      //     them, so a gateway that spreads a conversation across several
      //     Azure OpenAI resources rejects the replay with "The requested item
      //     was created under a different Azure OpenAI resource". The payload
      //     that rides along with the id is self-contained, so dropping the
      //     address leaves a request the provider accepts.
      //
      //     Only the listed providers are touched: a provider that owns a
      //     single resource validates the id itself and wants it back.
      if (opts.stripItemIds.length > 0) {
        const provider = (event.model as any)?.providerID
        if (typeof provider === "string" && opts.stripItemIds.includes(provider)) {
          for (const message of messages) {
            if (!Array.isArray(message?.content)) continue
            const kept = message.content.map((part: any) => stripItemId(part))
            if (kept.some((part: any, i: number) => part !== message.content[i])) {
              message.content = kept
            }
          }
        }
      }

      // The payload is the system prompt, the tool schemas and the messages.
      // The tool schemas are part of every request, so leaving them out
      // understates what was actually sent to the model.
      const toolsBytes = () =>
        Object.values((event.tools ?? {}) as Record<string, any>).reduce(
          (n: number, t: any) => n + bytes(t?.description ?? "") + bytes(JSON.stringify(t?.input ?? {})),
          0,
        )

      const before =
        (event.system ?? []).reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
        toolsBytes() +
        messages.reduce((n: number, m: any) => n + messageBytes(m), 0)

      // 2a. The budget pass.
      //
      //     One rule for every kind of content: the history is a byte stream,
      //     a budget says how much of the newest end is kept whole, and a
      //     quantized cut says where the kept region starts.
      //
      //     The cut is measured from the START of the history and rounded down
      //     to a step. The history only grows at the end, so the cut means the
      //     same thing on every request, and it only MOVES when a whole step
      //     has accumulated. Between moves the request is byte-identical and
      //     the provider's prompt cache still hits.
      //
      //     A sliding window would move the cut every turn, and every turn
      //     would be a full-price re-read of the whole history — which costs
      //     more than not trimming at all.
      let dropped = 0
      let droppedBytes = 0
      let historyElided = 0
      let historyBytes = 0
      let callsElided = 0
      let callsBytes = 0
      let textElided = 0
      let textBytes = 0
      let partIndex = 0

      const flat = flattenParts(messages)
      // The bands, per kind of content. Each kind gets its own byte stream
      // and its own cut, so a fat tool result never eats the reasoning
      // budget and vice versa.
      //
      // The bands are keyed by the part itself, not by its position. Two
      // independent walks of the same list would have to stay in step, and
      // the one thing this plugin must never do is trim the wrong part.
      const bands = new Map<any, Band>()
      for (const kind of ["reasoning", "tool-result", "tool-call", "text"] as const) {
        const ofKind = flat.filter((f) => f.part?.type === kind)
        const byIndex = new Map(ofKind.map((f) => [f.index, f.part]))
        for (const [index, band] of assignBands(ofKind, opts, kind)) {
          bands.set(byIndex.get(index), band)
        }
      }
      // A part carrying a provider signature is never touched: it has to be
      // replayed byte-identically or the provider rejects the request.
      for (const f of flat) {
        if (f.part?.type === "reasoning" && carriesSignature(f.part)) bands.set(f.part, "keep")
      }

      // Undo rewinds the session. A result that carried a marker can be back
      // in the kept region while the marker in the database still says
      // "elided" — so the marker is undone too, from the full text on disk.
      const restores = planRestores(flat, bands, opts)
      // Files this request still points at: the disk cap evicts them last.
      const referenced = new Set<string>()

      const ids: string[] = []
      for (const m of messages) if (typeof m?.id === "string") ids.push(m.id)
      for (const message of messages) {
        if (!Array.isArray(message?.content)) continue
        const kept: any[] = []
        let changed = false
        for (const part of message.content) {
          const index = partIndex++
          const type = part?.type
          const band = bands.get(part) ?? "keep"

          // -- reasoning ------------------------------------------------------
          if (type === "reasoning") {
            if (band === "keep" || carriesSignature(part)) {
              kept.push(part)
              continue
            }
            // The model has already acted on this reasoning. It is pure
            // overhead now, and it is the largest single source of payload
            // in a long session.
            //
            // Counted once. The same reasoning is re-sent on every request,
            // so counting it every time would report a saving the session
            // never had.
            if (ledger.claimElided(reasoningKey(message.id, part.text))) {
              dropped++
              droppedBytes += bytes(part.text ?? "")
            }
            changed = true
            continue
          }

          // -- tool results ---------------------------------------------------
          //
          //     Two tiers, and the difference matters:
          //
          //       - A result the model is still working with gets a DIGEST:
          //         its head and its tail, so the shape of the data survives
          //         and the model can still read it.
          //       - A result from long ago gets a POINTER: one line, naming
          //         the file the full text is in.
          //
          //     A session has hundreds of results, and a digest is not cheap:
          //     630 results digested at 4.6 KB each cost 863 KB, which is
          //     half of what the results themselves took. A pointer costs
          //     150 bytes, so all 630 cost 94 KB. The model has already acted
          //     on the old results, so what it needs is the knowledge that
          //     they existed and where to find them.
          if (type === "tool-result" && part.result) {
            // The part is replaced as the walk goes: images first, then the
            // text. One result can hold both, and both can be elided in the
            // same pass, so everything below reads `live`, not `part`.
            let live = part
            let result = part.result
            const key = part.id ?? `${message.id ?? "?"}#${index}`
            let text = resultText(part)
            let size = text === undefined ? 0 : bytes(text)
            // Pointers this walk created for images. The text replacement
            // below drops every text item after the first, and these are text
            // items that must survive.
            let imagePointers = new Set<any>()
            // Every file this request still points at is kept off the disk
            // cap's eviction list below.
            let marker = text === undefined ? undefined : markerPath(text)
            if (marker) referenced.add(marker)
            // Undo rewinds the session, so a result that was elided when it
            // was old can be the newest thing the model has again. The plan
            // was made newest first, inside the kept region and inside its
            // budget; here it is applied.
            const restore = restores.get(part)
            if (restore !== undefined) {
              kept.push(restorePart(part, restore))
              if (ledger.claimRestored(key)) {
                ledger.bump({ resultsRestored: 1, resultBytesRestored: bytes(restore) - size })
              }
              changed = true
              continue
            }
            // Images carry no text to digest, so every rule below walks past
            // them, and they are the biggest bytes in the history: one session
            // held 280 screenshots, 47 MB of a 48 MB payload. The item names
            // the file it was read from, so it becomes a line that says so.
            if (band !== "keep") {
              const images = elideImages(live, live.name ?? "result")
              if (images) {
                live = images.part
                result = live.result
                text = resultText(live)
                size = text === undefined ? 0 : bytes(text)
                if (ledger.claimElided(`${key}:img`)) {
                  historyElided++
                  historyBytes += images.saved
                }
                imagePointers = new Set(images.created)
                changed = true
              }
            }
            // A result this small is cheaper to keep than to replace: the
            // pointer alone is 150 bytes, and a 2 KB result is not worth
            // trading for one.
            const overFloor = size > opts.elideFloorBytes
            if (!text || !overFloor || band === "keep") {
              kept.push(live)
              continue
            }
            // The middle band is the one the model may still be working
            // with, so it keeps the head and the tail. The old band only
            // needs to say that the result existed and where to find it.
            //
            // If the pointer cannot be written — no disk, no room, the
            // content is not text — the result falls back to a digest rather
            // than to nothing. A digest costs bytes; losing the content
            // costs the model its ability to answer.
            const name = live.name ?? "result"
            // A text that already carries a marker IS a digest: its original
            // is on disk. Digesting it again costs the same bytes and points
            // at a file holding the digest instead of the original, so the
            // model would have to make two hops to get back to the data.
            const replacement =
              band === "elide"
                ? (elideToPointer(text, opts, name, size) ??
                  digestJson(text, opts, name) ??
                  digestResult(text, opts, name))
                : marker
                  ? undefined
                  : (digestJson(text, opts, name) ?? digestResult(text, opts, name))
            if (replacement === undefined) {
              kept.push(live)
              continue
            }
            const replacementFile = markerPath(replacement)
            if (replacementFile) referenced.add(replacementFile)
            // A replacement that is bigger than the original makes the
            // payload grow, which is the one thing the plugin must never do.
            if (bytes(replacement) >= size) {
              kept.push(live)
              continue
            }
            if (ledger.claimElided(key)) {
              historyElided++
              historyBytes += size - bytes(replacement)
            }
            if (result.type === "text") {
              kept.push({ ...live, result: { ...result, value: replacement } })
            } else {
              // The replacement goes into the FIRST text slot and the rest
              // are dropped. Writing it into every slot would multiply it:
              // a result split across 40 text items would go from 80 KB to
              // 40 x the replacement, and the payload would GROW.
              const value: any[] = []
              let placed = false
              for (const item of result.value) {
                if (item?.type !== "text") {
                  value.push(item)
                  continue
                }
                // An image pointer is already one line and says where the
                // bytes are. Dropping it would lose the only reference to a
                // screenshot the model may need to re-read.
                if (imagePointers.has(item)) {
                  value.push(item)
                  continue
                }
                if (placed) continue
                placed = true
                value.push({ type: "text", text: replacement })
              }
              kept.push({ ...live, result: { ...result, value } })
            }
            changed = true
            continue
          }

          // -- tool call arguments --------------------------------------------
          //
          //     The call itself stays: the model needs to know it made the
          //     call. Only the arguments are trimmed, and only the bulky
          //     string fields inside them. A `write` carries the whole file
          //     in its arguments, and a session that writes ten files replays
          //     all ten files on every turn forever.
          //
          //     A protected path is never touched, and that includes the
          //     ARGUMENTS of a call about a protected path. Trimming them
          //     would truncate the very edit an agent is making to a
          //     protected file, which is the one failure mode that leaves a
          //     session unable to recover.
          if (type === "tool-call" && part.input && band !== "keep") {
            const callPath = extractCallPath(part.input)
            const protectedCall =
              opts.protectTools.includes(part.name) ||
              (callPath !== undefined && isProtectedPath(callPath, opts.protectPaths))
            if (protectedCall) {
              if (ledger.claimElided(`p${index}`)) ledger.bump({ protectedSkipped: 1 })
              kept.push(part)
              continue
            }
            const trimmed = trimCallInput(part.input)
            if (trimmed.changed) {
              // Counted once, like the reasoning and the results. The same
              // argument is re-sent on every request, so counting it every
              // time reports a saving the session never had.
              if (ledger.claimElided(`c${index}`)) {
                callsElided++
                callsBytes += trimmed.saved
              }
              kept.push({ ...part, input: trimmed.input })
              changed = true
              continue
            }
          }

          // -- assistant text --------------------------------------------------
          //
          //     The model's own prose is the cheapest thing to keep and the
          //     most expensive to re-derive, so the budget is generous and
          //     what is cut keeps both ends: the head says what it was doing
          //     and the tail says what it concluded.
          if (type === "text" && band !== "keep") {
            const size = bytes(part.text ?? "")
            if (size > opts.elideFloorBytes) {
              // The excerpt is at most a third of the text, so the payload
              // always shrinks, and never more than the digest budgets allow.
              const excerpt = Math.min(Math.floor(size / 3), opts.headBytes + opts.tailBytes)
              const digest = trimText(part.text ?? "", excerpt)
              if (bytes(digest) < size) {
                if (ledger.claimElided(`t${index}`)) {
                  textElided++
                  textBytes += size - bytes(digest)
                }
                kept.push({ ...part, text: digest })
                changed = true
                continue
              }
            }
          }

          kept.push(part)
        }
        if (changed) message.content = kept
      }
      // The full-output directory is a cache, not an archive. One scan per
      // request is enough; the throttle inside the scan makes it one a minute.
      enforceDiskCap(opts.directory, opts.maxDiskBytes, referenced)
      ledger.retain(new Set(ids))
      ledger.bump({ reasoningDropped: dropped, reasoningBytesDropped: droppedBytes })
      ledger.bump({ resultsElided: historyElided, resultBytesDropped: historyBytes })
      ledger.bump({ callsElided, callBytesDropped: callsBytes })
      ledger.bump({ textElided, textBytesDropped: textBytes })

      // 2b. System prompt: cap it, cutting only at safe boundaries.
      let systemDropped = 0
      if (Array.isArray(event.system)) {
        const total = event.system.reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0)
        const budget = Math.max(0, opts.systemBytes - bytes(BRIEF))
        if (total > budget) {
          const parts = [...event.system]
          // Trim the largest parts first — the Code Mode catalog is the usual bulk.
          const order = parts
            .map((p: any, i: number) => ({ i, size: bytes(p.text ?? "") }))
            .sort((a, b) => b.size - a.size)
          let current = total
          for (const { i } of order) {
            if (current <= budget) break
            const part = parts[i]
            const size = bytes(part.text ?? "")
            // How much this part must give up to bring the total under budget.
            const excess = current - budget
            // Trim to the size the part must shrink TO, not by. Cutting by the
            // excess would drop the part to `excess` bytes instead of
            // `size - excess`, throwing away far more than the cap asks for.
            const target = Math.max(0, size - excess)
            const trimmed = trimSystemPart(part.text ?? "", target)
            if (trimmed === part.text) continue
            parts[i] = { ...part, text: trimmed }
            systemDropped += size - bytes(trimmed)
            current -= size - bytes(trimmed)
          }
          event.system = parts
        }
      }
      // The cap is re-applied from scratch on every request, so this is the
      // saving of the LAST request, not a running total. Adding it to a
      // cumulative total would count the same bytes once per turn.
      ledger.set({ systemBytesDropped: systemDropped })

      // 2c. Tool schemas: cap the combined budget.
      //
      //     Property descriptions are trimmed first — they repeat what the
      //     tool description already says, and dropping them cannot make a
      //     call malformed. Tool descriptions are trimmed last, and never
      //     below a floor, because they are the model's only guidance on when
      //     to use the tool. The JSON schema shape is always preserved.
      let toolDropped = 0
      if (event.tools && typeof event.tools === "object") {
        const entries = Object.entries(event.tools) as [string, any][]
        const sizeOf = (t: any) => bytes(t.description ?? "") + bytes(JSON.stringify(t.input ?? {}))
        let total = entries.reduce((n, [, t]) => n + sizeOf(t), 0)
        if (total > opts.toolBytes) {
          // Property descriptions, with a decreasing limit until the budget is met.
          for (const limit of [200, 100, 50, 20, 0]) {
            if (total <= opts.toolBytes) break
            for (const [, tool] of entries) {
              const before = bytes(JSON.stringify(tool.input ?? {}))
              const trimmed = trimSchemaDescriptions(tool.input, limit)
              const after = bytes(JSON.stringify(trimmed))
              if (after < before) {
                toolDropped += before - after
                total -= before - after
                tool.input = trimmed
              }
            }
          }
          // Then the tool descriptions, proportionally, down to a floor.
          if (total > opts.toolBytes) {
            const descTotal = entries.reduce((n, [, t]) => n + bytes(t.description ?? ""), 0)
            const schemaTotal = total - descTotal
            const allowed = Math.max(0, opts.toolBytes - schemaTotal)
            if (descTotal > allowed) {
              const scale = allowed / descTotal
              for (const [, tool] of entries) {
                const desc = tool.description ?? ""
                const size = bytes(desc)
                if (size === 0) continue
                const keep = Math.max(MIN_DESCRIPTION, Math.floor(size * scale))
                if (keep >= size) continue
                const trimmed = takeBytes(desc, keep)
                toolDropped += size - bytes(trimmed)
                total -= size - bytes(trimmed)
                tool.description = trimmed
              }
            }
          }
        }
      }
      // Same as the system prompt: this is the saving on the last request.
      ledger.set({ toolBytesDropped: toolDropped })

      // 2d. Measure, BEFORE the brief is added.
      //
      //     The brief is a fixed cost the plugin ADDS to every request. If it
      //     were included in the "after" number and not in the "before" one,
      //     the report would show the payload growing on every request, and a
      //     session that trimmed nothing would look like it had lost 20% to
      //     the brief. The two numbers have to be measured on the same basis,
      //     so the brief is excluded from both and reported on its own line.
      const after =
        (event.system ?? []).reduce((n: number, p: any) => n + bytes(p.text ?? ""), 0) +
        toolsBytes() +
        messages.reduce((n: number, m: any) => n + messageBytes(m), 0)
      ledger.bump({ requests: 1 })
      ledger.set({ requestBefore: before, requestAfter: after })
      // The peak is the largest payload this session would have sent with
      // nothing trimmed. The trimmed size of THAT SAME request is kept with
      // it: comparing the largest untrimmed payload against the most recent
      // trimmed one would mix two different requests and report a saving
      // that never happened.
      if (before > ledger.get().peakBefore) {
        ledger.set({ peakBefore: before, peakAfter: after })
      }

      // 2e. The behaviour brief. Constant bytes, so it does not disturb the
      //     cached system prefix. It is added AFTER the measurement above, so
      //     it is not counted as either a saving or a growth.
      if (Array.isArray(event.system) && !event.system.some((p: any) => (p.text ?? "").startsWith("## Context discipline"))) {
        event.system = [...event.system, { type: "text", text: BRIEF }]
        ledger.set({ briefBytes: bytes(BRIEF) })
      }
    })

    // -- 3. Stats command.
    //
    //     `resume: false` is essential. Without it, `synthetic` wakes the
    //     session's execution and the model answers the report: the numbers
    //     appear, and then the model is asked to talk about them, which costs
    //     a turn and a pile of tokens for something the user already sees.
    //     The built-in plan plugin passes the same flag for the same reason.
    const report = (sessionID: string) => {
      const s = ledgers.get(sessionID)?.get()
      if (!s || s.requests === 0) return "Token slim: no requests recorded yet."
      // Clamped at zero: a restore can make this request bigger than the
      // persisted history it came from, and "-12% smaller" is not a sentence.
      const pct = s.requestAfter > 0
        ? Math.max(0, Math.round((1 - s.requestAfter / Math.max(1, s.requestBefore)) * 100))
        : 0
      // The peak is compared against the trimmed size of the same request.
      const peakPct = s.peakAfter > 0
        ? Math.max(0, Math.round((1 - s.peakAfter / Math.max(1, s.peakBefore)) * 100))
        : 0
      const lines = [
        "Token slim — session savings",
        `requests: ${s.requests}`,
        "",
        "This request (trimmed, before the brief is added):",
        `  ${fmt(s.requestAfter)}  (${pct}% smaller than the untrimmed history)`,
      ]
      if (s.briefBytes > 0) {
        lines.push(
          "Actually sent, including the brief:",
          `  ${fmt(s.requestAfter + s.briefBytes)}`,
        )
      }
      lines.push(
        "",
        "Largest request this session, before and after:",
        `  ${fmt(s.peakBefore)} -> ${fmt(s.peakAfter)}  (${peakPct}% smaller)`,
        "",
        "Removed from the context, cumulatively:",
        `  reasoning:      ${fmt(s.reasoningBytesDropped)} (${s.reasoningDropped} parts)`,
        `  tool results:   ${fmt(s.resultBytesDropped)} (${s.resultsElided} results)`,
        `  call arguments: ${fmt(s.callBytesDropped)} (${s.callsElided} calls)`,
        `  own text:       ${fmt(s.textBytesDropped)} (${s.textElided} parts)`,
        `  duplicates:     ${fmt(s.duplicateBytesDropped)} (${s.duplicatesCollapsed} results)`,
      )
      if (s.protectedSkipped > 0) {
        lines.push(`  left alone:      ${s.protectedSkipped} (protected path or tool)`)
      }
      // A restore is the one thing that gives bytes back. Shown next to the
      // savings so the block above cannot be read as a net number.
      if (s.resultsRestored > 0) {
        lines.push(
          `  restored:       -${fmt(s.resultBytesRestored)} (${s.resultsRestored} results, back in the kept region)`,
        )
      }
      lines.push(
        "",
        "Kept whole, so the model does not lose its working context:",
        `  reasoning:      ${fmt(opts.reasoningBytes)} kept, ${fmt(opts.reasoningKeepBytes)} of it never cut`,
        `  tool results:   ${fmt(opts.resultKeepBytes)} kept, ${fmt(opts.resultDigestBytes)} more kept as head+tail digests`,
        `  call arguments: ${fmt(opts.callKeepBytes)} kept`,
        `  own text:       ${fmt(opts.textKeepBytes)} kept`,
      )
      lines.push(
        "",
        "Capped on every request (last request):",
        `  system prompt:  ${fmt(s.systemBytesDropped)}`,
        `  tool schemas:   ${fmt(s.toolBytesDropped)}`,
      )
      if (s.briefBytes > 0) {
        lines.push(`  brief added:    ${fmt(s.briefBytes)} (every request, in exchange for the waste it prevents)`)
      }
      if (s.snipCommands > 0) {
        lines.push(
          "",
          "Shell output filtered before it reached the context:",
          `  ${s.snipCommands} command(s), ${fmt(s.snipBytesFiltered)} of output sent to the model`,
          "  (snip keeps its own ledger: run `snip gain` for the bytes it removed)",
        )
      }
      const byTool = Object.entries(s.byTool).sort((a, b) => b[1] - a[1])
      if (byTool.length > 0) {
        lines.push("", "Where the savings came from:")
        for (const [tool, saved] of byTool.slice(0, 6)) {
          lines.push(`  ${tool.padEnd(14)} ${fmt(saved)}`)
        }
      }
      return lines.join("\n")
    }

    await ctx.command.transform((editor: any) => {
      editor.add({
        name: "slim-stats",
        description: "Show token-slim savings for this session",
        execute: async ({ sessionID }: any) => {
          const text = report(sessionID)
          // The TUI renders a synthetic message's description in a SINGLE
          // <text> element, which is one line. A multi-line report therefore
          // loses everything below the first line, and the user sees a title
          // and nothing else.
          //
          // So the report goes to a file, and the description carries the
          // one-line summary plus the path. The full numbers are one `cat`
          // away, and the TUI shows a line that actually fits.
          const file = writeReportFile(opts.statsDirectory, sessionID, text)
          const summary = reportSummary(text)
          await ctx.session.synthetic({
            sessionID,
            // The TUI renders a synthetic message's DESCRIPTION and skips one
            // with no description at all, so the report goes there.
            description: file ? `${summary}  (full report: ${file})` : summary,
            // The BODY is what enters the model's context if the message is
            // ever promoted, so it stays a one-liner. The report itself is
            // for the user, not for the model.
            text: "Token slim: the session savings report is shown above.",
            resume: false,
          })
        },
      })
    })

    // -- 4. The same report, for when the text arrives as a plain prompt.
    //
    //     The TUI dispatches a registered command itself, so it never gets
    //     here. The CLI does not: `opencode run "/slim-stats"` sends the
    //     text to the model as an ordinary prompt, and so does a TUI whose
    //     command list has not loaded yet.
    //
    //     The prompt is replaced by the report and the model is asked to
    //     print it as it stands. The model cannot be left out of it here:
    //     `opencode run` is a model turn by design, and a prompt hook cannot
    //     stop it. Failing the prompt would stop the model call, but it
    //     surfaces as a 500 and the user gets an error instead of the
    //     numbers, which is worse than one extra turn.
    await ctx.session.hook("prompt", async (event: any) => {
      const text = event?.prompt?.text
      if (typeof text !== "string") return
      if (!/^\/slim-stats(\s|$)/.test(text.trim())) return
      event.prompt.text = [
        report(event.sessionID),
        "",
        "Print the report above exactly as it is, with no commentary.",
      ].join("\n")
    })

    return () => {
      clearInterval(persistTimer)
      for (const [sessionID, ledger] of ledgers) {
        writePersistedStats(opts.statsDirectory, sessionID, ledger.get())
      }
    }
  },
}

// ---------------------------------------------------------------------------
// Helpers
// ---------------------------------------------------------------------------

/** A tool description shorter than this says less than it is worth keeping. */
export const MIN_DESCRIPTION = 240

/**
 * Anthropic-style thinking blocks carry a signature the provider validates on
 * replay. Such a part must be sent back byte-identical, so it is never touched.
 */
export function carriesSignature(part: any): boolean {
  if (typeof part.encrypted === "string" && part.encrypted.length > 0) return true
  const meta = part.providerMetadata
  if (!meta || typeof meta !== "object") return false
  for (const value of Object.values(meta)) {
    if (!value || typeof value !== "object") continue
    if ("signature" in value) return true
    if ("redactedData" in value) return true
  }
  return false
}

/**
 * Drop the provider's own item id from a part's metadata.
 *
 * The OpenAI Responses API replays the provider's item ids (`rs_...`,
 * `msg_...`, `fc_...`) alongside the history. Those ids are scoped to the
 * resource that minted them, so a gateway that load-balances a conversation
 * across several Azure OpenAI resources rejects the replay outright:
 * "The requested item was created under a different Azure OpenAI resource."
 *
 * The id is only an address. The reasoning payload that travels with it
 * (`reasoningEncryptedContent`) is self-contained, so removing the address
 * leaves a request the provider accepts, and the reasoning itself still
 * reaches the model.
 *
 * Returns the part unchanged when there is no id to drop.
 */
export function stripItemId(part: any): any {
  const meta = part?.providerMetadata
  if (!meta || typeof meta !== "object") return part
  let changed = false
  const next: Record<string, any> = {}
  for (const [key, value] of Object.entries(meta)) {
    if (!value || typeof value !== "object" || !("itemId" in value)) {
      next[key] = value
      continue
    }
    const { itemId: _dropped, ...rest } = value as Record<string, any>
    changed = true
    next[key] = rest
  }
  if (!changed) return part
  return { ...part, providerMetadata: next }
}

/**
 * The byte cost of one message, whether its content is a part list or a
 * plain string.
 *
 * The per-part cost is `partCost`, the same function the budget walk uses, so
 * the number reported and the number budgeted can never disagree. A string
 * content is counted rather than assumed away: it is legal in the frame, and
 * a measurement that throws takes the whole trim with it.
 */
function messageBytes(message: any): number {
  const content = message?.content
  if (Array.isArray(content)) return content.reduce((n: number, part: any) => n + partCost(part), 0)
  if (typeof content === "string") return bytes(content)
  return 0
}

/**
 * Cut a system part to a byte allowance without breaking the structure.
 *
 * The Code Mode catalog is a listing of tool signatures. Cutting it mid-entry
 * leaves the model with a half-written signature, so the cut lands on an entry
 * boundary and the tail is annotated with the way to find the rest. Prose
 * parts are cut on a line boundary.
 */
export function trimSystemPart(text: string, allowance: number) {
  if (bytes(text) <= allowance) return text
  if (allowance <= 0) return ""

  const isCatalog = text.includes("# Code Mode") || text.includes("<skill>")
  if (isCatalog) {
    const lines = text.split("\n")
    // A namespace header or a tool listing starts an entry.
    const startsEntry = (line: string) => /^-\s+\S/.test(line) || /^\s{2,}-\s+tools\./.test(line)
    let cut = -1
    for (let i = lines.length - 1; i >= 0; i--) {
      if (startsEntry(lines[i])) {
        cut = i
        break
      }
    }
    if (cut > 0) {
      const marker = "\n[truncated: this listing is incomplete. Use search() to find any tool that is not shown above.]"
      const kept = lines.slice(0, cut).join("\n")
      // The marker is part of the budget, not an extra on top of it.
      if (bytes(kept) + bytes(marker) <= allowance) {
        return `${kept}${marker}`
      }
      const room = allowance - bytes(marker)
      if (room > 0) {
        const head = takeBytes(kept, room)
        const atLine = head.lastIndexOf("\n")
        const safe = atLine > 0 ? head.slice(0, atLine) : head
        return `${safe}${marker}`
      }
    }
    // No usable boundary: keep the beginning, cut on a line boundary.
    const marker = "\n[truncated]"
    const room = allowance - bytes(marker)
    if (room <= 0) return ""
    const head = takeBytes(text, room)
    const atLine = head.lastIndexOf("\n")
    const safe = atLine > 0 ? head.slice(0, atLine) : head
    return `${safe}${marker}`
  }

  const head = takeBytes(text, allowance)
  const atLine = head.lastIndexOf("\n")
  return atLine > 0 ? head.slice(0, atLine) : head
}

/** Trim the `description` fields inside a JSON schema, never its shape. */
export function trimSchemaDescriptions(schema: any, limit: number): any {
  if (!schema || typeof schema !== "object") return schema
  if (Array.isArray(schema)) return schema.map((item) => trimSchemaDescriptions(item, limit))
  const out: any = {}
  for (const [key, value] of Object.entries(schema)) {
    if (key === "description" && typeof value === "string") out[key] = takeBytes(value, limit)
    else out[key] = trimSchemaDescriptions(value, limit)
  }
  return out
}

function fmt(n: number) {
  if (n < 1024) return `${n} B`
  if (n < 1024 * 1024) return `${(n / 1024).toFixed(1)} KB`
  return `${(n / 1024 / 1024).toFixed(2)} MB`
}
