/**
 * Regression test for the counter semantics.
 *
 * The `context` hook hands over a fresh, untrimmed copy of the history on
 * every request, so the same oversized tool result is re-elided on every
 * request. Counting it again each time inflated the reported saving by the
 * number of turns: a session could claim to have saved far more than it ever
 * sent.
 *
 * The system prompt and tool schema caps are re-applied from scratch every
 * request too, so they are reported for the last request, not accumulated.
 */
import assert from "node:assert/strict"
import { Ledger } from "./index.ts"

const bytes = (s: string) => Buffer.byteLength(s, "utf8")

// --- 1. a result elided on many requests is counted once ---------------------
{
  const ledger = new Ledger()
  const key = "prt_1"

  // Request 1: the result is new, so it is counted.
  assert.equal(ledger.claimElided(key), true, "the first request counts the result")

  // Requests 2..10: the same result comes back in the fresh history copy.
  let counted = 0
  for (let i = 0; i < 9; i++) {
    if (ledger.claimElided(key)) counted++
  }
  assert.equal(counted, 0, "the same result must not be counted again")
}

// --- 2. different results are each counted -----------------------------------
{
  const ledger = new Ledger()
  assert.equal(ledger.claimElided("a"), true)
  assert.equal(ledger.claimElided("b"), true)
  assert.equal(ledger.claimElided("a"), false)
  assert.equal(ledger.claimElided("b"), false)
}

// --- 3. the saved bytes match what was actually removed ----------------------
{
  const ledger = new Ledger()
  const big = "x".repeat(50_000)
  const digest = "short digest"

  let elided = 0
  let saved = 0
  // Ten requests, each re-eliding the same 50 KB result.
  for (let request = 0; request < 10; request++) {
    if (ledger.claimElided("prt_same")) {
      elided++
      saved += bytes(big) - bytes(digest)
    }
  }

  assert.equal(elided, 1, "one result, one count")
  assert.equal(saved, 50_000 - bytes(digest), "the saving is the real difference, not ten times it")
  // The counter must not claim more saving than the session ever held.
  assert.ok(saved < bytes(big), "the saving must be smaller than the result itself")
}

// --- 4. the per-request caps are not accumulated -----------------------------
{
  const ledger = new Ledger()
  // The system prompt cap is applied on every request.
  for (let request = 0; request < 5; request++) {
    ledger.set({ systemBytesDropped: 1000 })
    ledger.set({ toolBytesDropped: 500 })
  }
  const stats = ledger.get()
  assert.equal(stats.systemBytesDropped, 1000, "the system cap reports the last request, not the sum")
  assert.equal(stats.toolBytesDropped, 500, "the tool cap reports the last request, not the sum")
}

console.log("all checks passed")
