# wizstudio

Local studio for Ollama. One Python file, standard library only, styled with `~/Applications/themes/wiz-theme.css` (theme and accent picker in the header).

**Tools:** OCR · Translate · Prompt Generator · Roleplay · Rewrite · Chat · Caption · Extract

## Start / stop
```
wizstudio            # background on 0.0.0.0:9014  → http://127.0.0.1:9014
wizstudio --fg       # foreground
wizstudio-stop       # stop
wizstudio-cd         # cd here
```
Also available as the **wizstudio** card in AI Dock. Log: `~/.cache/wiz-studio/wiz-studio.log`.
Direct: `python3 wiz_studio.py --port 9014 [--ollama http://127.0.0.1:11434]`, self-test: `--self-test`.

## What you can do
Every tool has a live model list, temperature, a hide-`<think>` toggle, an editable system prompt, and a context-size field under Advanced. Output streams. Stop, Copy, Save, Preview (light markdown), word count, and **Send to** another tool. Finished OCR, translate, prompt, rewrite, caption, and extract runs land in the **Library** (search, reopen, delete).

| Tool | What it adds |
|---|---|
| OCR | Presets, language hint, multi-image and PDF pages (`pdftoppm`, up to 30), paste, remove a page |
| Translate | Tone, glossary (`source = target`), keep line breaks, one image |
| Prompt | Photoreal, Krea 2 (LoRA-aware), LoRA dataset shot with a trigger line, Qwen edit instruction, cinematic video, MiniMax H3 draft. Frame and variation count |
| Roleplay | Character browser with tags, eight diverse female starters, collapsible card editor, Enter-to-send, Continue, delete/edit/swipe replies, italic `*actions*`, attach image, import/export JSON (including SillyTavern) |
| Rewrite | Screenplay, dialogue, and subtext presets, plus a change view |
| Chat | Plain threads, separate from roleplay, with an optional image |
| Caption | Batch images into LoRA captions. Download a zip of sidecar `.txt` files |
| Extract | Read one image. Presets for camera and character position, outfit, jewellery, makeup, hair, pose, light, set, a prose description, an image-to-prompt paragraph, an H3 framing line, a Qwen edit inventory, or a full breakdown. LoRA-safe leaves face, hair, skin and body type out |

Shortcuts: `Ctrl+Enter` runs the open tool, `Esc` stops, `Alt+1`–`Alt+8` switches tools, `Ctrl+K` opens the library, `Ctrl+/` shows the list.

The H3 draft is a starting point. Production clips still go through H3 Prompter.

## Default models (change per tool, remembered)
| Tool | Default |
|---|---|
| OCR | unlimited-ocr-3B-Q8 |
| Translate | translategemma-4B-Q4KM |
| Prompt, Caption, Extract | qwen-3-vl-8B-Q8 |
| Roleplay | mag-mel-R1-12B-Q8 |
| Rewrite, Chat | gemma-4-7.5B-Q8 (the German learner preset switches to sauerkraut-8B-Q8) |

An older prompt preset named `Photoreal image` opens as `Photoreal still`. `MiniMax H3 structured` opens as `MiniMax H3 draft`.

OCR, Caption, and Extract only list models Ollama marks as vision. A saved text-only model is replaced with that tool's vision default. Attaching an image on any other tool does the same.

## Data
`~/.config/wiz-studio/`

- `settings.json` — per-tool model, temperature, system prompt, theme
- `characters.json`, `history/<character-id>.json`
- `threads.json`, `threads/<thread-id>.json`
- `library.json` — saved runs (text only, newest 200)

Settings are mirrored in browser localStorage under `wiz-studio-settings`.
