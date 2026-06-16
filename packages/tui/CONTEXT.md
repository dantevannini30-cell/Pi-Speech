# tui Domain Glossary

The Terminal UI framework — the interactive mode's rendering layer. Manages the terminal display, editor, keybindings, and visual components.

## TUI Core

- **tui.ts** (`src/tui.ts`): Main TUI class — manages screen rendering, component tree, event loop, and terminal lifecycle.
- **terminal.ts** (`src/terminal.ts`): Low-level terminal operations — raw mode, cursor control, colors, window resizing.
- **terminal-colors.ts**: Color palette and theme management.
- **terminal-image.ts**: Image rendering in the terminal (iTerm2/sixel protocols).

## Editor

- **editor-component.ts** (`src/editor-component.ts`): The text editor widget used for composing prompts. Handles text input, cursor movement, selection, scrolling.
- **kill-ring.ts**: Emacs-style kill ring for cut/copy/paste in the editor.
- **autocomplete.ts**: Tab-completion for commands and paths.
- **undo-stack.ts**: Undo/redo history for the editor.

## Keybindings

- **keybindings.ts** (`src/keybindings.ts`): Default keybinding definitions and the keybinding resolution system. Actions are mapped to key sequences (e.g. Ctrl+Space, Ctrl+O).
- **keys.ts**: Key event parsing and representation.
- **native-modifiers.ts**: macOS-specific modifier key handling.
- **word-navigation.ts**: Word-boundary movement logic.

## Components

- **components/** (`src/components/`): Individual UI widgets — input fields, chat display, status bar, help panel, etc.

## Input

- **stdin-buffer.ts**: Raw stdin buffering and event parsing.
- **fuzzy.ts**: Fuzzy matching for autocomplete/filtering.

## Utilities

- **utils.ts**: Shared TUI utilities — string formatting, layout helpers.
