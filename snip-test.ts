/**
 * Tests for the snip command repair.
 *
 * snip answers with the full command it wants run, and it leads with the
 * path of its own binary: `"C:\\Users\\me\\go\\bin\\snip.exe" run -- git log`.
 * That answer is broken on Windows: a quoted path is not a command in
 * PowerShell, and the path comes back JSON-escaped with doubled backslashes.
 *
 * The repair replaces that path with one the target shell can run. These
 * tests pin the rules, because a repair that is wrong is worse than no
 * repair: the command fails to parse and the agent has to run it again.
 */
import assert from "node:assert/strict"
import { repairSnipCommand } from "./index.ts"

// --- 1. the default: the bare name -----------------------------------------
{
  // snip's own path is dropped and the bare name is used instead. The plugin
  // spawns snip by bare name, so the name is on the PATH the shell inherits.
  // No quoting, no escaping, and it runs in cmd, PowerShell and bash alike.
  const out = repairSnipCommand('"C:\\\\Users\\\\me\\\\go\\\\bin\\\\snip.exe" run -- git log', "")
  assert.equal(out, "snip run -- git log")
}

// --- 2. an unquoted path is handled the same way ----------------------------
{
  const out = repairSnipCommand("/usr/local/bin/snip run -- git log", "")
  assert.equal(out, "snip run -- git log")
}

// --- 3. everything after snip's path is kept verbatim ----------------------
{
  // A future snip that adds a flag to `run` still works, because only the
  // leading path token is replaced.
  const out = repairSnipCommand('"C:\\\\x\\\\snip.exe" run --keep-errors -- git log', "")
  assert.equal(out, "snip run --keep-errors -- git log")
}

// --- 4. an explicit path without spaces stays unquoted ----------------------
{
  const out = repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', "C:\\tools\\snip.exe")
  assert.equal(out, "C:\\tools\\snip.exe run -- git log")
}

// --- 5. an explicit path with spaces needs quoting, and the shell decides ---
{
  const spaced = "C:\\Program Files\\snip.exe"
  // PowerShell needs the call operator to run a quoted path.
  assert.equal(
    repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', spaced, "powershell.exe"),
    '& "C:/Program Files/snip.exe" run -- git log',
  )
  assert.equal(
    repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', spaced, "pwsh.exe"),
    '& "C:/Program Files/snip.exe" run -- git log',
  )
  // cmd treats a leading `&` as a command separator and would run an empty
  // command first, so it must not have one.
  assert.equal(
    repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', spaced, "cmd.exe"),
    '"C:/Program Files/snip.exe" run -- git log',
  )
  // bash and friends do not need the call operator either.
  assert.equal(
    repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', spaced, "/bin/bash"),
    '"C:/Program Files/snip.exe" run -- git log',
  )
  // An unknown shell defaults to PowerShell, which is what OpenCode prefers
  // on Windows: pwsh, then powershell, then git bash, then cmd.
  assert.equal(
    repairSnipCommand('"C:\\\\x\\\\snip.exe" run -- git log', spaced),
    '& "C:/Program Files/snip.exe" run -- git log',
  )
}

// --- 6. a malformed answer is returned unchanged ----------------------------
{
  // No space and no closing quote: there is nothing to split, so the answer
  // is passed through rather than mangled.
  assert.equal(repairSnipCommand("snip", ""), "snip")
  assert.equal(repairSnipCommand('"C:\\\\x\\\\snip.exe', ""), '"C:\\\\x\\\\snip.exe')
}

console.log("all checks passed")
