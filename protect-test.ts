/**
 * Tests for the self-protection guard.
 *
 * A plugin that trims the agent's own context can trim the context the agent
 * is using to edit *this plugin*, which is how a session ends up writing a
 * broken edit back over a file it only half read. These tests pin the guard
 * that prevents it.
 */
import assert from "node:assert/strict"
import { fileURLToPath } from "node:url"
import { isProtectedPath, matchesPattern, extractPath } from "./index.ts"

// --- 1. glob matching -------------------------------------------------------
{
  // Exact match.
  assert.equal(matchesPattern("/a/b/c.ts", "/a/b/c.ts"), true)
  // A plain directory protects its whole subtree.
  assert.equal(matchesPattern("/a/b/c.ts", "/a/b"), true)
  assert.equal(matchesPattern("/a/b/c/d.ts", "/a/b"), true)
  // A sibling with a shared prefix is NOT protected.
  assert.equal(matchesPattern("/a/bc/d.ts", "/a/b"), false)
  // `*` stays inside a segment.
  assert.equal(matchesPattern("/a/b/c.ts", "/a/*/c.ts"), true)
  assert.equal(matchesPattern("/a/b/x/c.ts", "/a/*/c.ts"), false)
  // `**` crosses segments.
  assert.equal(matchesPattern("/a/b/x/c.ts", "/a/**/c.ts"), true)
  // Windows separators are normalised.
  assert.equal(matchesPattern("F:\\work\\a.ts", "F:/work"), true)
  // An empty pattern matches nothing.
  assert.equal(matchesPattern("/a/b", ""), false)
}

// --- 2. the plugin's own directory is always protected -----------------------
{
  // The plugin protects its own source by default, with no configuration.
  // This is the guarantee that matters: a session editing the plugin can
  // always read the plugin.
  // `fileURLToPath` is how the plugin derives the path, and the test must
  // derive it the same way: stripping "file://" by hand leaves a leading
  // slash on Windows ("/F:/..."), which is not the path anything else uses.
  const own = fileURLToPath(import.meta.url).replace(/\\/g, "/").replace(/\/[^/]*$/, "")
  assert.equal(isProtectedPath(own + "/index.ts", []), true)
  assert.equal(isProtectedPath(own, []), true)
  // An unrelated path is not protected by default.
  assert.equal(isProtectedPath("/somewhere/else/file.ts", []), false)
  // A configured path is protected on top of the default.
  assert.equal(isProtectedPath("/home/me/project/src/app.ts", ["/home/me/project"]), true)
  // A configured path that does not match stays unprotected.
  assert.equal(isProtectedPath("/home/me/other/app.ts", ["/home/me/project"]), false)
}

// --- 3. path extraction from a tool result -----------------------------------
{
  // OpenCode's read tool prefixes its output with the path it read.
  assert.equal(
    extractPath("Read file src/app.ts, lines 1-120\n1: export const a = 1"),
    "src/app.ts",
  )
  // A result with no path header yields nothing rather than a wrong guess.
  assert.equal(extractPath("just some output\nwith lines"), undefined)
  assert.equal(extractPath(""), undefined)
}

console.log("all checks passed")
