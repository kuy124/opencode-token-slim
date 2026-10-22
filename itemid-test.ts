/**
 * Regression test for the provider item id fix.
 *
 * The OpenAI Responses API replays the provider's own item ids with the
 * history. Those ids are addresses in the resource that minted them, so a
 * gateway that spreads a conversation across several Azure OpenAI resources
 * rejects the replay: "The requested item was created under a different Azure
 * OpenAI resource." The payload that rides along with the id is
 * self-contained, so dropping the address is enough.
 */
import assert from "node:assert/strict"
import plugin, { stripItemId } from "./index.ts"

// --- 1. the helper drops the id and keeps everything else ---------------------
{
  const part = {
    type: "reasoning",
    text: "thinking",
    providerMetadata: {
      openai: {
        itemId: "rs_abc123",
        reasoningEncryptedContent: "gAAAA...",
      },
    },
  }
  const out = stripItemId(part)
  assert.equal(out.providerMetadata.openai.itemId, undefined, "the item id must be gone")
  assert.equal(
    out.providerMetadata.openai.reasoningEncryptedContent,
    "gAAAA...",
    "the reasoning payload must survive",
  )
  assert.equal(out.text, "thinking", "the text must survive")
  assert.equal(part.providerMetadata.openai.itemId, "rs_abc123", "the input must not be mutated")
}

// --- 2. a part without an id is returned untouched ----------------------------
{
  const part = { type: "text", text: "hi", providerMetadata: { openai: { phase: "final_answer" } } }
  assert.equal(stripItemId(part), part, "a part with no id must be the same object")
  const bare = { type: "text", text: "hi" }
  assert.equal(stripItemId(bare), bare, "a part with no metadata must be the same object")
}

// --- 3. only the id goes, other metadata keys stay ---------------------------
{
  const part = {
    type: "text",
    text: "hi",
    providerMetadata: {
      openai: { itemId: "msg_abc", phase: "final_answer" },
      other: { keep: "me" },
    },
  }
  const out = stripItemId(part)
  assert.equal(out.providerMetadata.openai.phase, "final_answer")
  assert.deepEqual(out.providerMetadata.other, { keep: "me" }, "another provider's metadata must stay")
}

// --- 4. the hook strips ids only for the listed providers ---------------------
const mockCtx = (options: any) => {
  const hooks: Record<string, Function> = {}
  return {
    hooks,
    ctx: {
      options,
      tool: { hook: async () => {} },
      session: {
        hook: async (name: string, fn: Function) => {
          hooks[`session.${name}`] = fn
        },
        synthetic: async () => {},
      },
      command: { transform: async () => {} },
    },
  }
}

const eventWith = (providerID: string) => ({
  sessionID: "ses_test",
  model: { id: "gpt-6.1-sol", providerID },
  system: [],
  tools: {},
  messages: [
    {
      id: "m1",
      role: "assistant",
      content: [
        {
          type: "reasoning",
          text: "t",
          providerMetadata: { openai: { itemId: "rs_1", reasoningEncryptedContent: "enc" } },
        },
        {
          type: "text",
          text: "t",
          providerMetadata: { openai: { itemId: "msg_1" } },
        },
      ],
    },
  ],
})

{
  const { ctx, hooks } = mockCtx({ stripItemIds: ["agentrouter"] })
  await (plugin as any).setup(ctx)
  const event = eventWith("agentrouter")
  await hooks["session.context"](event)
  const parts = event.messages[0].content
  assert.equal(parts[0].providerMetadata.openai.itemId, undefined, "the listed provider's id must go")
  assert.equal(parts[1].providerMetadata.openai.itemId, undefined, "every id must go")
  assert.equal(
    parts[0].providerMetadata.openai.reasoningEncryptedContent,
    "enc",
    "the reasoning payload must survive",
  )
}

{
  const { ctx, hooks } = mockCtx({ stripItemIds: ["agentrouter"] })
  await (plugin as any).setup(ctx)
  const event = eventWith("openai")
  await hooks["session.context"](event)
  const parts = event.messages[0].content
  assert.equal(parts[0].providerMetadata.openai.itemId, "rs_1", "another provider's id must stay")
  assert.equal(parts[1].providerMetadata.openai.itemId, "msg_1", "another provider's id must stay")
}

{
  // The default is off: nothing is touched unless a provider is listed.
  const { ctx, hooks } = mockCtx({})
  await (plugin as any).setup(ctx)
  const event = eventWith("agentrouter")
  await hooks["session.context"](event)
  assert.equal(
    event.messages[0].content[0].providerMetadata.openai.itemId,
    "rs_1",
    "without configuration the ids must stay",
  )
}

console.log("all checks passed")
