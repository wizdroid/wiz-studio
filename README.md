# wizstudio

Local studio for Ollama. One Python file, standard library only, styled with `~/Applications/themes/wiz-theme.css` (theme and accent picker in the header).

**Tools:** OCR · Translate · Prompt Generator · Roleplay · Rewrite · Chat · Caption · Extract · T2VA · Ref2VA · Soundscape · Center Orbit · Character

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
| Rewrite | The same 18 modes and Light / Medium / Heavy strength as [wiz-text-rewriter](https://perchance.org/wiz-text-rewriter), a custom instruction that overrides the mode, plus studio extras (Screenplay, Tighten dialogue, Subtext, German learner) and a change view |
| Chat | Plain threads, separate from roleplay, with an optional image |
| Caption | Batch images into LoRA captions. Download a zip of sidecar `.txt` files |
| Extract | Read one image. Presets for camera and character position, outfit, jewellery, makeup, hair, pose, light, set, a prose description, an image-to-prompt paragraph, an H3 framing line, a Qwen edit inventory, or a full breakdown. LoRA-safe leaves face, hair, skin and body type out |
| T2VA | [wiz-t2va](https://perchance.org/wiz-t2va): MiniMax-H3 text-to-video clips as a reel. 20 movie types or auto, 6/8/10/15 s, direction, locked characters with Lock from scene, a location builder (category, place, side, era, time, weather, mood, style; AI generate, random, enhance, or instant without AI), Next scene / From input / Reset reel, and a no-AI fill-in if the model fails |
| Ref2VA | [wiz-ref2va](https://perchance.org/wiz-ref2va): reference-to-video prompt with subject definitions. People, places and voice references (9 images, 3 voices), interaction, duration, 13 dialogue languages |
| Soundscape | [wiz-soundscape](https://perchance.org/wiz-soundscape): overall_soundscape and non_diegetic_music fields. Music, genre, tempo, era, vocal, lyrics language, soundscape preset, 5–600 s clip |
| Center Orbit | [wiz-center-orbit](https://perchance.org/wiz-center-orbit): 360 room sweep from walls, corners and the middle. Sweep and six angles are built locally; AI can fill the form from a description or a room photo (vision model) |
| Character | [wiz-character](https://perchance.org/wiz-character): name, role, 17 genres, seed idea, 13 detail fields. Develop fills only empty fields, Deepen, per-field Regenerate, Build outputs (visual lock line, Krea2 / Qwen turnaround sheet, voice), Lock in T2VA |

Shortcuts: `Ctrl+Enter` runs the open tool, `Esc` stops, `Alt+1`–`Alt+0` switches tools (1–9, then 0 for the 10th), `Ctrl+K` opens the library, `Ctrl+/` shows the list.

The H3 draft is a starting point. Production clips still go through H3 Prompter.

## Default models (change per tool, remembered)
| Tool | Default |
|---|---|
| OCR | unlimited-ocr-3B-Q8 |
| Translate | translategemma-4B-Q4KM |
| Prompt, Caption, Extract, Center Orbit | qwen-3-vl-8B-Q8 |
| Roleplay | mag-mel-R1-12B-Q8 |
| Rewrite, Chat, T2VA, Ref2VA, Soundscape, Character | gemma-4-7.5B-Q8 (the German learner preset switches to sauerkraut-8B-Q8) |

An older prompt preset named `Photoreal image` opens as `Photoreal still`. `MiniMax H3 structured` opens as `MiniMax H3 draft`.

OCR, Caption, and Extract only list models Ollama marks as vision. A saved text-only model is replaced with that tool's vision default. Attaching an image on any other tool does the same.

## Data
`~/.config/wiz-studio/`

- `settings.json` — per-tool model, temperature, system prompt, theme
- `characters.json`, `history/<character-id>.json`
- `threads.json`, `threads/<thread-id>.json`
- `library.json` — saved runs (text only, newest 200)

Settings are mirrored in browser localStorage under `wiz-studio-settings`.

## Enhance / Generate

Every text tool has the same bar above the field it works on: **✨ Enhance** rewrites what you typed into a
richer, well-structured version that keeps your intent; **🎲 Generate** writes a complete input from a short
idea (or from scratch when the field is empty); **↶ Undo** restores the text from before. It uses the tab's
model and reads the tab's other settings (type, genre, duration, mode…) as context. Caption and Extract
have no bar: their input is the image and a preset.

All prompts live in one dict, `ASSIST_PROMPTS` (plus shared `ASSIST_RULES`) near the top of `wiz_studio.py`.
Tabs with several fields (Roleplay card, Ref2VA, Orbit, Character) answer with `KEY: value` lines, which
small local models follow more reliably than JSON. Prompt design draws on:

- OpenAI prompt-generation meta-prompt: https://developers.openai.com/api/docs/guides/prompt-generation
- Anthropic prompt engineering / prompt improver / metaprompt:
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/overview ,
  https://docs.anthropic.com/en/docs/build-with-claude/prompt-engineering/prompt-improver ,
  https://platform.claude.com/cookbook/misc-metaprompt
- Wan text-to-video prompts: https://www.alibabacloud.com/help/en/model-studio/text-to-video-prompt
- Veo prompt guide: https://deepmind.google/models/veo/prompt-guide/
- Suno style prompts: https://paddystudio.com/blog/suno-prompting-guide/
- Character turnaround sheets: https://arcloop.ai/handbook/en-US/character-turnaround-sheet-guide ,
  https://www.elser.ai/blog/ai-character-turnaround-prompts-front-side-back-views
