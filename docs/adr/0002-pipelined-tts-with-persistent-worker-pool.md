# Pipelined TTS with persistent worker pool

The TTS pipeline was processing sentences sequentially — synthesize, play, emit
done, then start the next sentence. The gap between sentences was the full
synthesis time of the next sentence (150-500ms with Piper, 1-3s with Kokoro).
The TypeScript side enforced this serialization with a promise chain
(`ttsSpeakChain`) that blocked each sentence on the previous one's full
playback completion.

We replaced the sequential path with a pipelined architecture: a pool of
persistent Python subprocess workers that use a two-phase protocol internally
(`speak` → `ready` → `play` → `done`) so sentence N+1 generates audio while
sentence N plays. The TS side sends sentences fire-and-forget; the Python
worker pool handles ordering and concurrency.

Status: proposed

## Consequences

- Zero audible gap between sentences (generation hides behind playback).
- The TS side drops the promise chain entirely — `TTSService.speak()` writes
  one JSON line and returns immediately. A separate `flush()` method provides a
  promise that resolves when all queued sentences finish playing.
- `tts_worker.py` moves from a synchronous stdin loop to asyncio, using
  `StreamingKokoroTTS` / `StreamingPiperTTS` which already exist.
- Worker pool size is configurable via `pool_size` in TTS settings (default 2).
- Speed changes and pausing work dynamically per sentence since each `speak`
  message includes the current speed.
