"""Entry point for whisper-bot.

Run::

    python -m whisper_bot
    # or
    ./whisper-bot
"""

from __future__ import annotations

import asyncio
import os
import sys


async def main() -> None:
    """Main entry point — runs dependency checks and starts the pipeline."""
    debug_enabled = "--debug" in sys.argv
    raw_enabled = "--raw" in sys.argv
    simple_mode = "--simple" in sys.argv
    from whisper_bot.debug import set_enabled as _set_debug
    from whisper_bot.debug import set_file as _set_debug_file

    _set_debug(debug_enabled)

    from whisper_bot.debug import log as _debug

    # In TUI mode, redirect debug output to a file so it doesn't get
    # swallowed by Textual's terminal management.
    if debug_enabled and not simple_mode:
        debug_path = os.path.expanduser("~/.whisper-bot/debug.log")
        _set_debug_file(debug_path)
        _debug(f"[DEBUG __main__] TUI debug output -> {debug_path}")

    _debug("[DEBUG __main__] Starting whisper-bot...")
    if raw_enabled:
        _debug("[DEBUG __main__] --raw flag set: skipping LLM parser, passing raw STT to agent")
    if simple_mode:
        _debug("[DEBUG __main__] --simple flag set: using stdin-based CLI mode")

    if simple_mode:
        # Legacy stdin-based mode (no TUI)
        print("whisper-bot 0.1.0")
        print()
        from whisper_bot.pipeline import run_pipeline

        try:
            await run_pipeline(skip_parser=raw_enabled)
        except KeyboardInterrupt:
            print()
            _debug("[DEBUG __main__] KeyboardInterrupt received")
        except Exception as exc:
            # CancelledError is a BaseException, so it passes through
            # Exception clause.  ExceptionGroup from TaskGroup cleanup
            # is also caught here.
            if isinstance(exc, asyncio.CancelledError):
                _debug("[DEBUG __main__] CancelledError caught (normal shutdown)")
                return
            if isinstance(exc, ExceptionGroup):
                non_cancelled = [e for e in exc.exceptions if not isinstance(e, asyncio.CancelledError)]
                if non_cancelled:
                    _debug(f"[DEBUG __main__] ExceptionGroup with non-cancelled errors: {non_cancelled[0]}")
                    print(f"Fatal error: {non_cancelled[0]}", file=sys.stderr)
                    sys.exit(1)
                _debug("[DEBUG __main__] ExceptionGroup with only cancelled errors (normal shutdown)")
                return
            _debug(f"[DEBUG __main__] Unhandled exception: {exc}")
            print(f"Fatal error: {exc}", file=sys.stderr)
            sys.exit(1)
    else:
        # Textual TUI mode (default)
        from whisper_bot.tui import WhisperBotApp

        app = WhisperBotApp(skip_parser=raw_enabled)
        await app.run_async()


if __name__ == "__main__":
    asyncio.run(main())
