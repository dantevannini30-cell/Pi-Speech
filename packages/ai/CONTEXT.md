# ai Domain Glossary

The AI/LLM layer — model definitions, provider implementations, and the API registry that the agent uses to communicate with LLMs.

## Models

- **models.ts** (`src/models.ts`): Defines the canonical model representations — capability flags (streaming, thinking, vision, function calling), context window sizes, pricing, and model families.
- **models.generated.ts** (`src/models.generated.ts`): Auto-generated model metadata. Do not edit directly — update `scripts/generate-models.ts` and regenerate.
- **image-models.ts**, **image-models.generated.ts**: Image generation model definitions (parallel structure to text models).

## Providers

- **providers/** (`src/providers/`): Each provider (Anthropic, OpenAI, Google, DeepSeek, Ollama, etc.) has its own subdirectory with implementation, types, and tests.
- **api-registry.ts** (`src/api-registry.ts`): Lazy registration system — providers register themselves with the registry, which the agent queries to pick the right provider for a given model.
- **types.ts** (`src/types.ts`): Core provider types — Provider interface, ModelConfig, Message, ToolCall, etc.

## Streaming

- **stream.ts** (`src/stream.ts`): SSE/streaming abstractions for consuming LLM responses incrementally.

## Configuration

- **env-api-keys.ts**: API key resolution from environment variables.
- **cli.ts**: CLI model listing, provider selection.
- **oauth.ts**: OAuth flow for providers that require it (e.g., Google AI Studio, GitHub Models).
- **session-resources.ts**: Session-scoped resource management for provider connections.

## Generated Models

- **scripts/generate-models.ts**: Script that scrapes provider model catalogs and regenerates `models.generated.ts`. Run when adding new models or syncing provider updates.
