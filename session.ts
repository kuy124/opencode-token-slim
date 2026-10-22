/**
 * The OpenCode database, and the session the dev scripts replay.
 *
 * No session id is written in the source. A session id belongs to one machine
 * and one conversation, and a script that carries one is a script nobody else
 * can run — and a leaked id in a repository. The default is the longest
 * session in the database, which is the one worth measuring.
 */
import { DatabaseSync } from "node:sqlite"
import os from "node:os"
import path from "node:path"

export const dbPath = path.join(
  process.env.XDG_DATA_HOME || path.join(os.homedir(), ".local", "share"),
  "opencode",
  "opencode.db",
)

export function largestSession(db: DatabaseSync): string {
  const row = db
    .prepare(
      "SELECT session_id AS id, COUNT(*) AS n FROM session_message GROUP BY session_id ORDER BY n DESC LIMIT 1",
    )
    .get() as { id?: string } | undefined
  if (!row?.id) throw new Error("no sessions in the database — pass a session id as an argument")
  return String(row.id)
}
