# agent Domain Glossary

The core agent loop — the runtime that drives model interactions, tool execution, and message handling. This is the orchestration layer that connects the LLM to everything else.

## Agent Loop

- **agent-loop.ts** (`src/agent-loop.ts`): The main execution loop — sends messages to the model, processes tool call responses, manages conversation history, and handles errors. Drives the turn-by-turn interaction.
- **agent.ts** (`src/agent.ts`): Agent configuration and lifecycle — model selection, system prompt construction, tool registration, and output handling.

## Types

- **types.ts** (`src/types.ts`): Core agent types — AgentConfig, ConversationMessage, ToolResult, ToolUse, etc.

## Harness

- **harness/** (`src/harness/`): Test harness utilities for running the agent in controlled environments (used in tests, not production).

## Node

- **node.ts** (`src/node.ts`): Node.js-specific agent setup and initialization.

## Proxy

- **proxy.ts** (`src/proxy.ts`): Proxy configuration for agent HTTP requests (useful in corporate/restricted environments).

## Message Flow

1. User sends a message → agent-loop formats it with conversation history
2. Model responds (text and/or tool calls)
3. Tool calls are dispatched to registered tools
4. Tool results are fed back to the model
5. Loop continues until the model produces a final text response or hits limits
