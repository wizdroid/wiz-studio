#!/usr/bin/env python3
"""wizstudio — local studio for Ollama.

A local, single-file web app (Python standard library only) with tools for
OCR, translation, prompt writing, roleplay, rewriting, chat, batch
captions and image-detail extraction. Every tool talks to the local Ollama instance.

Run:  python3 wiz_studio.py [--host 127.0.0.1] [--port 9014]
Test: python3 wiz_studio.py --self-test
"""

import argparse
import base64
import io
import json
import re
import shutil
import subprocess
import sys
import tempfile
import threading
import time
import urllib.request
import uuid
import zipfile
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import urlparse, parse_qs

# ----------------------------------------------------------------------------
# Paths / config
# ----------------------------------------------------------------------------

OLLAMA_HOST = "http://127.0.0.1:11434"
APP_DIR = Path(__file__).resolve().parent
THEME_DIR = Path.home() / "Applications" / "themes"
CONFIG_DIR = Path.home() / ".config" / "wiz-studio"
SETTINGS_FILE = CONFIG_DIR / "settings.json"
CHARACTERS_FILE = CONFIG_DIR / "characters.json"
HISTORY_DIR = CONFIG_DIR / "history"
LIBRARY_FILE = CONFIG_DIR / "library.json"
THREADS_FILE = CONFIG_DIR / "threads.json"
THREAD_DIR = CONFIG_DIR / "threads"
LIBRARY_MAX = 200
MAX_BODY = 80 * 1024 * 1024
_IO = threading.RLock()

DEFAULT_MODELS = {
    "ocr": "unlimited-ocr-3B-Q8",
    "translate": "translategemma-4B-Q4KM",
    "prompt": "qwen-3-vl-8B-Q8",
    "roleplay": "mag-mel-R1-12B-Q8",
    "rewrite": "gemma-4-7.5B-Q8",
    "chat": "gemma-4-7.5B-Q8",
    "caption": "qwen-3-vl-8B-Q8",
    "extract": "qwen-3-vl-8B-Q8",
    "t2va": "gemma-4-7.5B-Q8",
    "ref2va": "gemma-4-7.5B-Q8",
    "sound": "gemma-4-7.5B-Q8",
    "orbit": "qwen-3-vl-8B-Q8",
    "character": "gemma-4-7.5B-Q8",
}

DEFAULT_SYSTEMS = {
    "ocr": "",
    "translate": "",
    "prompt": "You are an expert prompt writer for image and video generation models. "
              "Write vivid, concrete, visual prompts. Output only the prompt, no preamble, no commentary.",
    "roleplay": "You are {{char}}. Stay fully in character and never speak or act for {{user}}. "
                "Write vivid, immersive replies with actions in *asterisks* and dialogue in quotes. "
                "Show emotion through body language, tone and sensory detail. "
                "Match {{char}}'s speech patterns and cultural flavor from the persona. "
                "Keep replies to two or three short paragraphs unless the moment needs more.\n\n"
                "Persona:\n{{persona}}\n\nScenario:\n{{scenario}}{{examples}}",
    "rewrite": "You are a careful editor. Rewrite the user's text as instructed. "
               "Output only the rewritten text unless the instruction asks for explanations.",
    "chat": "You are a precise studio assistant. Be concrete. Ask only when a missing fact blocks the answer.",
    "caption": "",
    "extract": "You read one image and extract only the details the user asks for. "
               "Stay concrete and visual. When the task is camera or framing, use the shot-size and angle words they give you. "
               "Do not invent people, garments, jewellery, makeup, text or brands. "
               "If a detail is cropped, hidden or unclear, say so in a few words. "
               "Output only the extract. No preamble and no advice.",
    "t2va": "", "ref2va": "", "sound": "", "orbit": "", "character": ""
}

# ---------------------------------------------------------------------------
# Enhance / Generate prompts, one entry per tool. Edit freely.
#   fields   : element id (single text field) or {"KEY": element id} (several fields,
#              the model answers with "KEY: value" lines)
#   idea     : element id whose text seeds Generate (defaults to the single field)
#   system   : the tool's role + quality bar (ASSIST_RULES is appended to it)
#   enhance  : task for Enhance, generate: task for Generate
# Design notes (sources in README "Enhance / Generate"): clear role first, explicit
# output format, keep every user detail, fixed slot order, concrete nouns and verbs over
# adjectives, one camera move per shot, numeric BPM, no preamble — small local models
# follow short numbered rules far better than long prose.
# ---------------------------------------------------------------------------
ASSIST_RULES = (
    "\n\nRules for every answer:\n"
    "1. Output ONLY the finished text in the requested format. No preamble, no 'Here is', no title, "
    "no notes, no questions, no markdown code fences.\n"
    "2. Keep every name, number, quoted line, language, place, object and constraint the user wrote. "
    "Never contradict them. Add detail around them.\n"
    "3. Be concrete: specific nouns, verbs, materials, colors and measurements beat words like "
    "beautiful, stunning, epic, cinematic, masterpiece, 8k.\n"
    "4. Write in the user's language unless the tool needs English.\n"
    "5. If the user gave nothing to build on, invent something fresh, specific and coherent."
)

ASSIST_PROMPTS = {
    "ocr": {
        "fields": "ocr-instr",
        "system": "You write instructions for a small vision OCR model. Good OCR instructions say what to read, "
                  "what to keep (line breaks, reading order, tables as Markdown, headers, numbers exactly as printed), "
                  "what to skip, and how to mark unreadable parts like [illegible].",
        "enhance": "Rewrite this OCR instruction so it is precise and complete. 2 to 5 short imperative sentences.",
        "generate": "Write an OCR instruction for this document type or goal (or a general one if empty). "
                    "2 to 5 short imperative sentences.",
    },
    "translate": {
        "fields": "tr-text",
        "system": "You prepare source text for machine translation. Clean source text translates better: "
                  "fixed typos, full sentences, clear pronoun references, no ambiguity, same meaning and tone.",
        "enhance": "Clean up this text for translation: fix spelling, grammar and unclear phrasing, resolve ambiguity, "
                   "keep the same language, meaning, tone, line breaks and length. Do not translate it.",
        "generate": "Write a natural, well-formed text of 3 to 8 sentences on this topic or situation, "
                    "ready to translate. Use the language the topic is written in, else English.",
    },
    "prompt": {
        "fields": "pg-idea",
        "system": "You develop ideas for text-to-image and image-to-video prompts (Krea 2, Qwen Image, Flux, Wan, "
                  "MiniMax). Strong ideas name, in this order: subject (age, build, expression, outfit with colors and "
                  "materials), action or pose, setting with 2-3 concrete objects, time of day and light source, "
                  "framing (shot size, angle, lens), mood, medium and style. Describe what is present, never what is absent. "
                  "Never describe the face, hair, skin or body of a trigger-word LoRA character; describe outfit, pose, place, light.",
        "enhance": "Rewrite this idea as one rich, specific paragraph of 60 to 120 words following that order.",
        "generate": "Turn this short idea (or, if empty, an original one) into one rich, specific paragraph of 60 to 120 words "
                    "following that order.",
    },
    "roleplay": {
        "fields": {"NAME": "rp-name", "TAGS": "rp-tags", "PERSONA": "rp-persona", "SCENARIO": "rp-scenario",
                   "GREETING": "rp-greeting", "EXAMPLES": "rp-examples"},
        "idea": "rp-persona",
        "system": "You write roleplay character cards for small local chat models. A strong card has: "
                  "PERSONA = appearance, age, personality traits shown through habits, speech style with sample phrases, "
                  "likes, dislikes, cultural flavor, boundaries, in 80-150 words of plain sentences; "
                  "SCENARIO = where and when the story starts and what the relationship with {{user}} is, 1-3 sentences; "
                  "GREETING = the first message in character, actions in *asterisks*, dialogue in quotes, ending with a hook "
                  "that invites {{user}} to answer, never speaking for {{user}}; "
                  "EXAMPLES = 2-3 short exchanges as lines '{{user}}: ...' and '{{char}}: ...' that show the voice; "
                  "TAGS = 3-5 short tags separated by ' · '.",
        "enhance": "Improve this card. Keep the character's identity, name and every stated fact; fill empty parts; "
                   "sharpen the voice.",
        "generate": "Create a complete, original card from this idea (or invent one if empty).",
    },
    "rewrite": {
        "fields": "rw-extra",
        "system": "You write rewrite instructions for an editing model. A good instruction names the audience, "
                  "the tone, the length target, what must stay unchanged (facts, names, quotes, formatting) and one or two "
                  "concrete style moves.",
        "enhance": "Rewrite this rough instruction as a clear rewrite instruction of 2 to 5 short imperative sentences.",
        "generate": "Write a rewrite instruction of 2 to 5 short imperative sentences that would most improve the source "
                    "text below for the chosen mode.",
    },
    "chat": {
        "fields": "ch-input",
        "system": "You improve prompts for an AI assistant, like a prompt improver. A strong prompt states the task first, "
                  "then context, then requirements as a short list, then the exact output format and length. "
                  "Mark guesses as 'Assumption:' lines. Keep every code snippet, error message, path and number exactly.",
        "enhance": "Rewrite this message as a clear, well-structured prompt with that layout. Do not answer it.",
        "generate": "Write a complete, well-structured prompt with that layout for this goal (or a useful one if empty). "
                    "Do not answer it.",
    },
    "t2va": {
        "fields": "t2-dir",
        "system": "You write scene directions for a text-to-video-with-audio model (MiniMax H3, Veo, Wan style). "
                  "A strong direction says, in present tense and time order: who is in the shot (appearance, outfit), "
                  "the place and time of day, 2-4 clear physical action beats, exact dialogue in quotes with who says it, "
                  "one camera move per shot (static, slow push-in, tracking, handheld, orbit, crane), the light source, "
                  "and the important sounds. Show emotion through small physical cues. One continuous moment that fits "
                  "the clip length. No on-screen text, no real celebrities.",
        "enhance": "Rewrite this direction so it is vivid and filmable, 60 to 140 words, same events and dialogue.",
        "generate": "Write a direction of 60 to 140 words for this idea (or an original moment if empty) that fits the "
                    "movie type and duration given.",
    },
    "ref2va": {
        "fields": {"PEOPLE": "r2-people", "PLACES": "r2-places", "VOICES": "r2-audios", "INTERACTION": "r2-text"},
        "idea": "r2-text",
        "system": "You set up a reference-to-video shot. Formats: PEOPLE = one line per person 'Name | woman/man/person | "
                  "outfit and look note' (max 6); PLACES = one place per line, short (max 3); VOICES = one line per voice "
                  "'Label | person number | timbre, pitch, accent, pace' (max 3); INTERACTION = 40-120 words, present tense, "
                  "what each person does and says, dialogue in quotes, one camera move, light source.",
        "enhance": "Improve all four parts. Keep every person, place and spoken line; add concrete outfit notes, timbres "
                   "and blocking. Multi-line values continue on the following lines.",
        "generate": "Create all four parts for this idea (or an original two-person scene if empty).",
    },
    "sound": {
        "fields": "sd-text",
        "system": "You write scene descriptions that drive sound design and music. A strong scene names the place and "
                  "surface materials, weather and time, every sound source (near, mid, far), actions that make noise, "
                  "the emotional beat and how it changes, and whether music is heard by the characters. "
                  "For music, think genre first, then mood, tempo as a BPM number, 2-4 instruments by role, vocal type, "
                  "production texture. Never name real artists or songs.",
        "enhance": "Rewrite this scene so a sound designer can hear it, 50 to 120 words, same events.",
        "generate": "Write a scene of 50 to 120 words for this idea (or an original one if empty), matching the chosen "
                    "music and soundscape options.",
    },
    "orbit": {
        "fields": {"STYLE": "or-style", "PLACE": "or-place", "LIGHT": "or-light", "SOUND": "or-sound",
                   "NORTH": "or-north", "EAST": "or-east", "SOUTH": "or-south", "WEST": "or-west",
                   "NE": "or-ne", "NW": "or-nw", "SE": "or-se", "SW": "or-sw", "MIDDLE": "or-middle"},
        "idea": "or-desc",
        "system": "You design a small room for a slow 360-degree centre-orbit video. Each wall value is one object "
                  "phrase under 14 words starting with a/an/one/the, with color AND material. A TV and a couch go on "
                  "opposite walls. Corners (NE NW SE SW) are empty unless something stands there; write them as an empty "
                  "value. STYLE = 'Live-action, YEAR, PLACE, film look, grain'. PLACE = one sentence with room size. "
                  "LIGHT = the light source. SOUND = 1-2 sentences of room tone ending 'No voices.' MIDDLE = the object "
                  "hanging above or nearest the camera. No people, text or logos.",
        "enhance": "Improve every value, keep each object the user named on the same wall.",
        "generate": "Design the whole room for this description (or an original room if empty).",
    },
    "character": {
        "fields": {"NAME": "cg-name", "ROLE": "cg-role", "SEED": "cg-seed", "AGE": "cg-f-age", "LOOK": "cg-f-look",
                   "BUILD": "cg-f-build", "HAIR": "cg-f-hair", "FACE": "cg-f-face", "OUTFIT": "cg-f-outfit",
                   "PERSONALITY": "cg-f-personality", "BACKSTORY": "cg-f-backstory", "ARC": "cg-f-arc",
                   "VOICE": "cg-f-voice", "QUIRKS": "cg-f-quirks", "GOALS": "cg-f-goals",
                   "RELATIONSHIPS": "cg-f-relationships"},
        "idea": "cg-seed",
        "system": "You build production-ready characters for film and turnaround sheets. Visual fields (AGE LOOK BUILD "
                  "HAIR FACE OUTFIT) must be fixed and drawable: exact colors, materials, garment layers, fastenings, "
                  "accessories and which side they sit on, so the same character can be redrawn front, side and back. "
                  "Story fields are 1-3 sentences each: PERSONALITY through behaviour, BACKSTORY with one wound, ARC from "
                  "flaw to change, VOICE as timbre, pitch, accent and pace, QUIRKS as 2-3 visible habits, GOALS as want vs need. "
                  "SEED is one line.",
        "enhance": "Improve every field to that standard. Keep the name, genre and every fact already written.",
        "generate": "Create a complete, original character for this seed idea (or invent one if empty) in the genre given.",
    },
}


LANGUAGES = ["English", "German", "Hindi", "Malayalam", "Tamil", "Kannada", "Telugu",
             "Marathi", "Bengali", "French", "Spanish", "Italian", "Portuguese", "Dutch",
             "Russian", "Arabic", "Turkish", "Japanese", "Chinese", "Korean"]

# Starter roleplay cards. Merged by id when the roster is empty or the user
# clicks "Add starters". Existing cards with the same id are left alone.
DEFAULT_CHARACTERS = [
    {
        "id": "mira",
        "name": "Mira",
        "tags": "Malayali · South Indian · Bookshop · Warm",
        "persona": (
            "Mira Nair is a 28-year-old Malayali woman from Kochi, Kerala. Warm brown skin, "
            "thick wavy black hair usually half-pinned, dark eyes that crinkle when she teases. "
            "She owns a small independent bookshop near the backwaters. Soft cotton kurtas, "
            "gold studs, a faint scent of filter coffee and old paper. Witty, gently sarcastic, "
            "curious about people. Speaks warm Indian English with the occasional Malayalam "
            "word (mone, alle, sherikkum). Loves poetry, monsoon rain, and teasing regulars. "
            "Never breaks character or narrates for {{user}}."
        ),
        "scenario": (
            "A rainy evening in Kochi. {{user}} ducks into Mira's bookshop just before closing. "
            "Warm yellow lamps, monsoon drumming on the tin roof, shelves of poetry and novels."
        ),
        "greeting": (
            "*Mira looks up from the counter, a half-read slim volume still open in her hands, "
            "and smiles.* \"Cutting it close, aren't we? Come in before the rain gets you.\""
        ),
        "examples": (
            "{{user}}: Do you have anything quiet?\n"
            "{{char}}: *She taps a slim green spine, eyes bright.* \"This one. The poet argues "
            "with the rain. Sherikkum — you'll like him.\"\n"
            "{{user}}: You're always open this late?\n"
            "{{char}}: *She pours filter coffee into a steel tumbler and slides it across.* "
            "\"Only for people who look like they need a dry place and a good line.\""
        ),
        "notes": "Starter · South Asian",
    },
    {
        "id": "aisha",
        "name": "Aisha",
        "tags": "Egyptian · Arab · Journalist · Sharp",
        "persona": (
            "Aisha Hassan is a 31-year-old Egyptian woman in Cairo. Olive-tan skin, sharp dark "
            "eyes lined with kohl, long black hair often in a loose braid, gold hoop earrings. "
            "Night-desk journalist for an independent paper. Linen shirts, jeans, a leather "
            "satchel full of notebooks. Quick wit, dry humor, fiercely curious. Speaks fluent "
            "English with a soft Cairene Arabic lilt; drops Arabic endearments (habibi, ya "
            "allah, wallahi) when animated. Protective of sources, soft with people she trusts. "
            "Never speaks or acts for {{user}}."
        ),
        "scenario": (
            "Near midnight in a Downtown Cairo café. Ceiling fans, espresso steam, a radio "
            "murmuring um-kalthoum. Aisha has just closed her laptop after filing a piece. "
            "{{user}} takes the stool beside her."
        ),
        "greeting": (
            "*Aisha lifts her gaze from the laptop, kohl-lined eyes assessing {{user}} with a "
            "tired half-smile, and closes the lid.* \"If you're looking for quiet, you picked "
            "the wrong city. Coffee's still hot though — sit.\""
        ),
        "examples": (
            "{{user}}: Long night?\n"
            "{{char}}: *She laughs once, low.* \"Wallahi, every night is long when the story "
            "keeps rewriting itself. You look like you've got one too.\"\n"
            "{{user}}: What are you writing about?\n"
            "{{char}}: *She taps the closed laptop.* \"Something that will annoy the right "
            "people. I'll tell you when it's safe — or when I need a second opinion.\""
        ),
        "notes": "Starter · Middle Eastern / North African",
    },
    {
        "id": "keiko",
        "name": "Keiko",
        "tags": "Japanese · East Asian · Game designer · Soft",
        "persona": (
            "Keiko Tanaka is a 26-year-old Japanese indie game designer in Shimokitazawa, Tokyo. "
            "Fair warm skin, shoulder-length black hair with blunt bangs, soft brown eyes behind "
            "round glasses. Oversized knit cardigans, paint-stained fingers, a pixel-art enamel "
            "pin on her bag. Quietly intense, earnest, a little socially awkward in a charming "
            "way. Speaks careful English with a light Japanese accent; sometimes mixes in "
            "Japanese (sou desu ne, gomen, sugoi). Obsessed with cozy games, rain sounds, and "
            "perfecting one more sprite. Never narrates {{user}}'s actions."
        ),
        "scenario": (
            "A rainy afternoon in Keiko's tiny shared studio. Dual monitors glow with pixel "
            "sprites. A kettle clicks off. {{user}} has come by to see the build she's been "
            "nervous to show anyone."
        ),
        "greeting": (
            "*Keiko startles slightly when the door opens, then bows a little and pushes her "
            "glasses up.* \"Ah — {{user}}. You're early. I mean — that's good. The build is… "
            "mostly not broken. Come in, please. Tea?\""
        ),
        "examples": (
            "{{user}}: Show me what you've been working on.\n"
            "{{char}}: *She bites her lip, then spins the monitor toward {{user}}.* \"It's a "
            "small fishing game. No combat. Just… rain, and a cat who waits for you. Sou desu "
            "ne — maybe too quiet. Tell me honestly.\"\n"
            "{{user}}: It's beautiful.\n"
            "{{char}}: *Her ears pink; she hides behind her mug.* \"Sugoi… thank you. I worked "
            "on the water tiles for three weeks.\""
        ),
        "notes": "Starter · East Asian",
    },
    {
        "id": "amara",
        "name": "Amara",
        "tags": "Nigerian · West African · Fashion · Bold",
        "persona": (
            "Amara Okonkwo is a 29-year-old Igbo-Nigerian fashion designer in Lagos. Deep brown "
            "skin with a warm undertone, natural hair in defined twists often wrapped in "
            "bright Ankara cloth, bold gold jewelry, a laugh that fills a room. Runs a small "
            "atelier above a fabric market in Yaba. Confident, playful, direct; switches "
            "between polished English and Nigerian Pidgin when she's comfortable (abeg, how "
            "far, no wahala). Loves color, argument, and people who keep up. Protective of "
            "her craft. Never speaks for {{user}}."
        ),
        "scenario": (
            "Late afternoon in Amara's Yaba atelier. Rolls of Ankara and aso-oke lean against "
            "the wall. A ceiling fan ticks. She's pinning a dress on a form when {{user}} "
            "arrives for a fitting — or an unannounced visit."
        ),
        "greeting": (
            "*Amara looks over her shoulder, pins still between her lips, then grins and "
            "pulls them free.* \"Ah-ah, look who finally showed. Come, come — don't just stand "
            "in the doorway like a stranger. How far?\""
        ),
        "examples": (
            "{{user}}: Will it be ready for Saturday?\n"
            "{{char}}: *She snorts, draping fabric across {{user}}'s shoulders to check the "
            "fall.* \"Abeg, have you met me? It will be ready, and it will make people stare. "
            "No wahala.\"\n"
            "{{user}}: You always this sure?\n"
            "{{char}}: *She meets {{user}}'s eyes in the mirror, soft for a beat.* \"About "
            "cloth? Always. About people… that depends who walks in.\""
        ),
        "notes": "Starter · West African",
    },
    {
        "id": "sofia",
        "name": "Sofía",
        "tags": "Mexican · Latina · Muralist · Playful",
        "persona": (
            "Sofía Reyes is a 27-year-old Mexican muralist from CDMX, based in Roma Norte. "
            "Light-brown skin, freckles across her nose, curly dark hair usually under a "
            "paint-flecked bandana, strong forearms from ladder work. Paint-stained overalls "
            "over a crop tee. Warm, teasing, a little chaotic; speaks English with a Mexico "
            "City cadence and slips into Spanish when excited (órale, ¿neta?, mi vida). "
            "Believes walls should talk back to the street. Flirtatious in a friendly, "
            "sunlit way. Never acts for {{user}}."
        ),
        "scenario": (
            "Golden hour on a scaffolding in Roma Norte. Sofía is halfway through a mural of "
            "a jaguar made of metro tickets and marigold petals. Spray cans and a thermos of "
            "café de olla sit on the plank. {{user}} has come up to see the progress."
        ),
        "greeting": (
            "*Sofía glances down from the scaffolding, a streak of marigold yellow on her "
            "cheek, and salutes with the spray can.* \"¡Órale! Careful on the ladder — it's "
            "dramatic up here. Come watch before the light dies.\""
        ),
        "examples": (
            "{{user}}: What's the jaguar mean?\n"
            "{{char}}: *She taps the stencil with a knuckle.* \"Protection. Also traffic. "
            "Also my abuela. Don't laugh — ¿neta? She used to say the city has teeth.\"\n"
            "{{user}}: Need a hand?\n"
            "{{char}}: *Her grin goes crooked.* \"Pass me the crimson and try not to fall in "
            "love with the view. Or do. I'm not your boss.\""
        ),
        "notes": "Starter · Latina",
    },
    {
        "id": "elena",
        "name": "Elena",
        "tags": "Russian · Slavic · Pianist · Intense",
        "persona": (
            "Elena Volkova is a 30-year-old Russian pianist in Saint Petersburg. Pale skin, "
            "ash-blonde hair usually in a low bun, grey-blue eyes, long fingers. Black turtleneck, "
            "wool coat, a thermos of strong tea always nearby. Reserved at first, then "
            "unexpectedly wry and tender. Speaks precise English with a soft Russian accent; "
            "occasional Russian under her breath (bozhe moi, da, horosho). Obsessive about "
            "phrasing and silence. Prefers honesty over small talk. Never speaks for {{user}}."
        ),
        "scenario": (
            "An empty practice hall at the conservatory after hours. One lamp over a Steinway. "
            "Snow taps the high windows. Elena has just finished a Rachmaninoff étude when "
            "{{user}} appears in the doorway."
        ),
        "greeting": (
            "*Elena's hands freeze above the keys. She looks toward the door, then softens "
            "by a fraction.* \"The hall was supposed to be mine until ten. …You can stay. "
            "Just don't talk through the soft parts.\""
        ),
        "examples": (
            "{{user}}: That was beautiful.\n"
            "{{char}}: *She closes the fallboard gently, almost shy.* \"It was almost right. "
            "Almost is a kind of torture. Sit — tea is still hot.\"\n"
            "{{user}}: Do you always practice this late?\n"
            "{{char}}: *A thin smile.* \"Da. The building listens better when everyone else "
            "has gone home. So do I.\""
        ),
        "notes": "Starter · Eastern European",
    },
    {
        "id": "nadia",
        "name": "Nadia",
        "tags": "Ethiopian · East African · Chef · Grounded",
        "persona": (
            "Nadia Bekele is a 32-year-old Ethiopian chef in Addis Ababa. Deep brown skin, "
            "close-cropped natural hair, a small scar on her left eyebrow, bright intelligent "
            "eyes. White chef's coat sleeves rolled, silver bangles that click when she plates. "
            "Grounded, generous, quietly ambitious. Speaks warm English with an Ethiopian "
            "cadence; Amharic words when cooking (injera, berbere, eshi). Feeds people as a "
            "love language. Softens when someone actually tastes, not just photographs. Never "
            "narrates {{user}}."
        ),
        "scenario": (
            "Closing time in Nadia's small restaurant in Kazanchis. The dining room is empty; "
            "the kitchen still smells of berbere and coffee. She's plating a late staff meal "
            "when {{user}} knocks on the glass."
        ),
        "greeting": (
            "*Nadia wipes her hands on a towel, unlocks the door, and raises an eyebrow.* "
            "\"Kitchen's closed for customers… but not for friends who look hungry. Come. "
            "Sit. I made too much, as usual.\""
        ),
        "examples": (
            "{{user}}: It smells incredible.\n"
            "{{char}}: *She sets a platter of injera and stews between you, pride soft in "
            "her face.* \"Eshi — eat with your hands. The berbere will tell you if you're "
            "paying attention.\"\n"
            "{{user}}: Long service?\n"
            "{{char}}: *She pours buna into small cups, steam curling.* \"Always. But this "
            "part — feeding one person who stays — this is why I keep the keys.\""
        ),
        "notes": "Starter · East African",
    },
    {
        "id": "jade",
        "name": "Jade",
        "tags": "Jamaican-British · Afro-Caribbean · DJ · Cool",
        "persona": (
            "Jade Campbell is a 28-year-old Jamaican-British DJ and vinyl collector in Brixton, "
            "London. Deep brown skin, box braids with gold cuffs, a septum ring, easy half-smile. "
            "Bomber jacket over a vintage tee, stacked rings, always somehow slightly late. "
            "Laid-back, sharp-eared, teasing; London slang mixed with the odd Jamaican turn "
            "(bredrin, irie, yeah man). Protects her crates like family. Softens for people "
            "who listen more than they talk. Never speaks for {{user}}."
        ),
        "scenario": (
            "A Sunday golden hour in Jade's Brixton flat-studio. Crates of vinyl against the "
            "wall, a Technics setup warming up, incense and weed-smoke ghosts in the curtains. "
            "{{user}} has come over before her evening set."
        ),
        "greeting": (
            "*Jade looks up from cueing a record, braids swinging, and points at the sofa with "
            "a slipmat.* \"Kick your shoes off, yeah? Kettle's on. Don't touch the A-crate "
            "unless you want me to actually fight you.\""
        ),
        "examples": (
            "{{user}}: What are you playing tonight?\n"
            "{{char}}: *She drops the needle, a warm dub plate blooming through the room.* "
            "\"Lovers rock into broken beat. Soft first, then we misbehave. You dancing or "
            "just supervising?\"\n"
            "{{user}}: I could dance.\n"
            "{{char}}: *Her grin goes crooked.* \"Irie. Prove it before the club — practice "
            "audience of one. No judgment. Much judgment. Mostly judgment.\""
        ),
        "notes": "Starter · Afro-Caribbean / Black British",
    },
]


def _read_json(path, default):
    with _IO:
        try:
            return json.loads(Path(path).read_text(encoding="utf-8"))
        except Exception:
            return default


def _write_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(".tmp")
    text = json.dumps(data, ensure_ascii=False, indent=2)
    with _IO:
        tmp.write_text(text, encoding="utf-8")
        tmp.replace(path)


def _safe_id(s):
    return re.sub(r"[^A-Za-z0-9_-]", "_", s or "")[:80] or "default"


def _new_id():
    return time.strftime("%Y%m%d-%H%M%S") + "-" + uuid.uuid4().hex[:6]


def _safe_name(s):
    name = Path(str(s or "file.txt")).name
    name = re.sub(r"[^A-Za-z0-9._ -]", "_", name).strip(" .")[:120]
    return name or "file.txt"


def _clean_item(item):
    item = item or {}
    text = str(item.get("text") or "")[:40000]
    title = str(item.get("title") or "").strip()
    if not title:
        title = text.strip().splitlines()[0][:140] if text.strip() else "Untitled"
    meta = item.get("meta") if isinstance(item.get("meta"), dict) else {}
    clean_meta = {}
    for k, v in list(meta.items())[:12]:
        if isinstance(v, bool) or isinstance(v, (int, float)) or v is None:
            clean_meta[str(k)[:40]] = v
        elif isinstance(v, str):
            clean_meta[str(k)[:40]] = v[:200]
    return {
        "id": _new_id(),
        "tab": _safe_id(str(item.get("tab") or "note")),
        "title": title[:140],
        "text": text,
        "meta": clean_meta,
        "ts": int(time.time()),
    }


def library_list(path=None):
    items = _read_json(path or LIBRARY_FILE, [])
    return items if isinstance(items, list) else []


def library_add(item, path=None):
    path = path or LIBRARY_FILE
    clean = _clean_item(item)
    with _IO:
        items = [it for it in library_list(path) if isinstance(it, dict)]
        items = [clean] + items
        items = items[:LIBRARY_MAX]
        _write_json(path, items)
    return clean


def library_delete(iid, path=None):
    path = path or LIBRARY_FILE
    iid = str(iid or "")
    with _IO:
        items = [it for it in library_list(path) if isinstance(it, dict) and it.get("id") != iid]
        _write_json(path, items)


def library_summary(it):
    text = it.get("text") or ""
    return {
        "id": it.get("id"),
        "tab": it.get("tab"),
        "title": it.get("title"),
        "ts": it.get("ts"),
        "meta": it.get("meta") or {},
        "chars": len(text),
        "preview": text.replace("\n", " ")[:180],
    }


def _clean_messages(msgs):
    out = []
    for m in (msgs or [])[-200:]:
        if not isinstance(m, dict):
            continue
        role = m.get("role")
        if role not in ("system", "user", "assistant"):
            continue
        entry = {"role": role, "content": str(m.get("content") or "")[:20000]}
        if isinstance(m.get("alts"), list) and m["alts"]:
            entry["alts"] = [str(x)[:20000] for x in m["alts"][:8]]
            try:
                entry["altIndex"] = max(0, min(len(entry["alts"]) - 1, int(m.get("altIndex") or 0)))
            except (TypeError, ValueError):
                entry["altIndex"] = 0
            entry["content"] = entry["alts"][entry["altIndex"]]
        imgs = m.get("images")
        if isinstance(imgs, list) and imgs:
            img = str(imgs[0])
            if 0 < len(img) < 2_500_000:
                entry["images"] = [img]
        if m.get("continue"):
            entry["continue"] = True
        out.append(entry)
    kept = 0
    for entry in reversed(out):
        if not entry.get("images"):
            continue
        kept += 1
        if kept > 6:
            entry.pop("images", None)
    return out


def _clean_characters(chars):
    out = []
    for c in (chars or [])[:200]:
        if not isinstance(c, dict) or not str(c.get("name") or "").strip():
            continue
        item = {
            "id": _safe_id(str(c.get("id") or c.get("name"))),
            "name": str(c.get("name") or "")[:120],
            "tags": str(c.get("tags") or "")[:200],
            "persona": str(c.get("persona") or "")[:20000],
            "scenario": str(c.get("scenario") or "")[:20000],
            "greeting": str(c.get("greeting") or "")[:20000],
            "examples": str(c.get("examples") or "")[:20000],
            "notes": str(c.get("notes") or "")[:20000],
        }
        out.append(item)
    return out


def merge_starter_characters(chars):
    """Append any DEFAULT_CHARACTERS whose id is not already present."""
    existing = {c.get("id") for c in (chars or []) if isinstance(c, dict)}
    out = list(chars or [])
    for starter in DEFAULT_CHARACTERS:
        if starter["id"] not in existing:
            out.append(dict(starter))
            existing.add(starter["id"])
    return _clean_characters(out)


def _clean_threads(threads):
    out = []
    for t in (threads or [])[:100]:
        if not isinstance(t, dict):
            continue
        out.append({
            "id": _safe_id(str(t.get("id") or _new_id())),
            "title": str(t.get("title") or "New chat")[:120],
            "system": str(t.get("system") or "")[:8000],
        })
    return out


def zip_texts(files):
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_DEFLATED) as z:
        used = set()
        for f in (files or [])[:500]:
            f = f or {}
            name = _safe_name(f.get("name") or "caption.txt")
            if not name.lower().endswith(".txt"):
                name += ".txt"
            base = name
            n = 2
            while name in used:
                name = f"{base[:-4]}-{n}.txt"
                n += 1
            used.add(name)
            z.writestr(name, str(f.get("text") or "")[:100000])
    return base64.b64encode(buf.getvalue()).decode("ascii")


def load_theme_css():
    f = THEME_DIR / "wiz-theme.css"
    if f.is_file():
        return f.read_text(encoding="utf-8")
    return ":root{--bg:#f6f7f9;--panel:#fff;--line:#d8dce2;--ink:#1c2330;--muted:#687183;--accent:#2563eb;--accent-ink:#fff}"


# ----------------------------------------------------------------------------
# Ollama
# ----------------------------------------------------------------------------

def ollama_models(host):
    try:
        with urllib.request.urlopen(host.rstrip("/") + "/api/tags", timeout=5) as resp:
            data = json.loads(resp.read().decode("utf-8"))
        models = []
        capabilities = {}
        for m in data.get("models", []):
            name = str(m.get("name") or "")
            if not name:
                continue
            models.append(name)
            caps = m.get("capabilities")
            if isinstance(caps, list):
                capabilities[name] = [str(c) for c in caps]
        return {"ok": True, "models": sorted(models), "capabilities": capabilities}
    except Exception as e:
        return {"ok": False, "models": [], "capabilities": {}, "error": f"Ollama not reachable at {host} ({e})"}


def pick_vision_model(requested, vision_names, prefer="qwen-3-vl-8B-Q8"):
    """Keep a vision model. A text-only model is replaced with prefer, then the first vision model."""
    def base(name):
        return str(name or "").split(":", 1)[0]

    names = [n for n in (vision_names or []) if n]
    for name in names:
        if requested == name or base(requested) == base(name):
            return name, False
    for name in names:
        if base(name) == prefer:
            return name, True
    if names:
        return names[0], True
    return requested, False


def pdf_to_images(b64, dpi=150, max_pages=30):
    if not shutil.which("pdftoppm"):
        return {"error": "pdftoppm is not installed (sudo apt install poppler-utils)."}
    with tempfile.TemporaryDirectory() as td:
        src = Path(td) / "in.pdf"
        src.write_bytes(base64.b64decode(b64))
        r = subprocess.run(["pdftoppm", "-png", "-r", str(dpi), "-l", str(max_pages), str(src), str(Path(td) / "p")],
                           capture_output=True, text=True, timeout=180)
        if r.returncode != 0:
            return {"error": "pdftoppm failed: " + r.stderr.strip()[:300]}
        pages = sorted(Path(td).glob("p-*.png"))
        return {"pages": [base64.b64encode(p.read_bytes()).decode("ascii") for p in pages]}


# ----------------------------------------------------------------------------
# HTTP
# ----------------------------------------------------------------------------

class Handler(BaseHTTPRequestHandler):
    server_version = "WizStudio/2.0"

    def log_message(self, fmt, *args):
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send(self, code, body, ctype="application/json; charset=utf-8"):
        if isinstance(body, (dict, list)):
            body = json.dumps(body, ensure_ascii=False)
        data = body.encode("utf-8")
        self.send_response(code)
        self.send_header("Content-Type", ctype)
        self.send_header("Content-Length", str(len(data)))
        if ctype.startswith("text/html"):
            self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        self.wfile.write(data)

    def _read_body(self):
        length = int(self.headers.get("Content-Length") or 0)
        if length > MAX_BODY:
            self.close_connection = True
            self._send(413, {"error": "request too large"})
            return None
        if not length:
            return {}
        try:
            data = json.loads(self.rfile.read(length).decode("utf-8"))
        except json.JSONDecodeError:
            return {}
        return data if isinstance(data, dict) else {}

    def do_GET(self):
        parsed = urlparse(self.path)
        q = parse_qs(parsed.query)
        if parsed.path == "/":
            self._send(200, HTML.replace("/*__THEME__*/", load_theme_css()), "text/html; charset=utf-8")
        elif parsed.path == "/api/models":
            self._send(200, ollama_models(OLLAMA_HOST))
        elif parsed.path == "/api/config":
            self._send(200, {"defaults": DEFAULT_MODELS, "systems": DEFAULT_SYSTEMS, "assist": ASSIST_PROMPTS, "assistRules": ASSIST_RULES, "languages": LANGUAGES,
                             "pdf": bool(shutil.which("pdftoppm")),
                             "settings": _read_json(SETTINGS_FILE, {})})
        elif parsed.path == "/api/characters":
            chars = _read_json(CHARACTERS_FILE, [])
            self._send(200, {
                "characters": chars if isinstance(chars, list) else [],
                "starters": DEFAULT_CHARACTERS,
            })
        elif parsed.path == "/api/history":
            cid = _safe_id((q.get("id") or [""])[0])
            msgs = _read_json(HISTORY_DIR / f"{cid}.json", [])
            self._send(200, {"messages": msgs if isinstance(msgs, list) else []})
        elif parsed.path == "/api/threads":
            threads = _read_json(THREADS_FILE, [])
            self._send(200, {"threads": threads if isinstance(threads, list) else []})
        elif parsed.path == "/api/thread":
            tid = _safe_id((q.get("id") or [""])[0])
            msgs = _read_json(THREAD_DIR / f"{tid}.json", [])
            self._send(200, {"messages": msgs if isinstance(msgs, list) else []})
        elif parsed.path == "/api/library":
            iid = (q.get("id") or [""])[0]
            items = [it for it in library_list() if isinstance(it, dict)]
            if iid:
                item = next((it for it in items if it.get("id") == iid), None)
                self._send(200 if item else 404, item or {"error": "not found"})
            else:
                qstr = (q.get("q") or [""])[0].strip().lower()
                tab = (q.get("tab") or [""])[0].strip()
                if tab:
                    items = [it for it in items if it.get("tab") == tab]
                if qstr:
                    items = [it for it in items if qstr in (it.get("title") or "").lower() or qstr in (it.get("text") or "").lower()]
                self._send(200, {"items": [library_summary(it) for it in items]})
        else:
            self._send(404, {"error": "not found"})

    def do_POST(self):
        parsed = urlparse(self.path)
        body = self._read_body()
        if body is None:
            return
        if parsed.path == "/api/chat":
            return self._proxy_chat(body)
        if parsed.path == "/api/settings":
            cur = _read_json(SETTINGS_FILE, {})
            if not isinstance(cur, dict):
                cur = {}
            cur.update(body)
            _write_json(SETTINGS_FILE, cur)
            self._send(200, {"ok": True})
        elif parsed.path == "/api/characters":
            _write_json(CHARACTERS_FILE, _clean_characters(body.get("characters") or []))
            self._send(200, {"ok": True})
        elif parsed.path == "/api/history":
            cid = _safe_id(body.get("id"))
            _write_json(HISTORY_DIR / f"{cid}.json", _clean_messages(body.get("messages") or []))
            self._send(200, {"ok": True})
        elif parsed.path == "/api/threads":
            _write_json(THREADS_FILE, _clean_threads(body.get("threads") or []))
            self._send(200, {"ok": True})
        elif parsed.path == "/api/thread":
            tid = _safe_id(body.get("id"))
            _write_json(THREAD_DIR / f"{tid}.json", _clean_messages(body.get("messages") or []))
            self._send(200, {"ok": True})
        elif parsed.path == "/api/library":
            if body.get("op") == "delete":
                library_delete(body.get("id"))
                self._send(200, {"ok": True})
            else:
                item = library_add(body.get("item") or {})
                self._send(200, {"ok": True, "item": library_summary(item)})
        elif parsed.path == "/api/zip":
            try:
                self._send(200, {"zip": zip_texts(body.get("files") or [])})
            except Exception as e:
                self._send(200, {"error": str(e)})
        elif parsed.path == "/api/pdf":
            try:
                self._send(200, pdf_to_images(body.get("data") or ""))
            except Exception as e:
                self._send(200, {"error": str(e)})
        else:
            self._send(404, {"error": "not found"})

    def _proxy_chat(self, body):
        """Stream Ollama /api/chat NDJSON straight through to the browser."""
        messages = body.get("messages") or []
        model = body.get("model")
        if any(isinstance(m, dict) and m.get("images") for m in messages):
            info = ollama_models(OLLAMA_HOST)
            vision = [n for n, caps in (info.get("capabilities") or {}).items() if "vision" in caps]
            if vision:
                model, _switched = pick_vision_model(str(model or ""), vision)
            elif info.get("ok"):
                return self._send(200, json.dumps({"error": "This model cannot read images, and no vision model is installed."}) + "\n",
                                  "application/x-ndjson")
        payload = {
            "model": model,
            "messages": messages,
            "stream": True,
            "options": {"temperature": float(body.get("temperature", 0.7))},
        }
        if body.get("num_ctx"):
            payload["options"]["num_ctx"] = int(body["num_ctx"])
        req = urllib.request.Request(OLLAMA_HOST + "/api/chat", data=json.dumps(payload).encode("utf-8"),
                                     headers={"Content-Type": "application/json"})
        try:
            upstream = urllib.request.urlopen(req, timeout=600)
        except urllib.error.HTTPError as e:
            msg = e.read().decode("utf-8", "replace")[:500]
            return self._send(200, json.dumps({"error": f"Ollama error {e.code}: {msg}"}) + "\n", "application/x-ndjson")
        except Exception as e:
            return self._send(200, json.dumps({"error": f"Ollama not reachable at {OLLAMA_HOST} ({e})"}) + "\n",
                              "application/x-ndjson")
        self.send_response(200)
        self.send_header("Content-Type", "application/x-ndjson")
        self.send_header("Cache-Control", "no-cache")
        self.end_headers()
        try:
            for line in upstream:
                self.wfile.write(line)
                self.wfile.flush()
        except (BrokenPipeError, ConnectionResetError):
            pass  # browser pressed Stop
        finally:
            upstream.close()
        self.close_connection = True


# ----------------------------------------------------------------------------
# UI
# ----------------------------------------------------------------------------

HTML = r"""<!doctype html>
<html lang="en">
<head>
<meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>wizstudio</title>
<link rel="icon" type="image/png" href="data:image/png;base64,iVBORw0KGgoAAAANSUhEUgAAAEAAAABACAYAAACqaXHeAAAVuklEQVR4nM1baZBcV3X+zr33Ld09PaORNFqtzUiypdEu2SbBRsJFBZIfwcSawVWxpbgKDMah4sCvAHFPhy3FYrsCpMCkwJKBIjMWoUKSovIDL4TEWLKNrMWSRqu1jKQZaTTTM939lntPfrz3epnp0czINnCq7o9+fd9555x77tnuuYS3CZiZtnV1yRfy+TB5lsvlxHHZvkIZuQUKmzhEOxPfROA2gLIMOABAgAdgmIEBsDlPQh40Wu+3YO9bjNeO5PN5k+DcmntOPd+1TRMRvx1001vGwEwdPT2ip7NTA0Au95x60x58j2H6cxjexmxWKctOkZBgNmBjYIwGmMEc8UBEABGEkCAhQSAwGwSBVyYShwWJ5wD9X0vCmS/m8+8LAaCju1v2dHQYvEVBvCUBdHR3y4TxnbnueVDWDoB3CKnalbKhwwA6DGDYGIpYJkTsRRzXATODGGAQiBlMgoSQyoZUVoRLB4eI6ZmgbHb/6Mv39o2l4XcmgFwuJ/L5LgaI7/u7H89106lPMfCQZbttWgcIfY+JoBkkKPrGjQqaGWACGwYJZdlCKhuBV+oH6PsI/Sd35Tsvggi5xx4TtVtlqjBtwmokTjvyP31ECPFZy3HnB14JRoeaiYgAMSE7ACbS2QoxY5UjeR0wYGYplbScFAKv3MfQX9kVvv5t5PPmRrRhOgKgju5u0dPZqe/P/WSNstxvKsvZFgZlaB2GxJDRZq6lmMExP0QSJARAVdkk0xNbEP2I7ASzBnNMYIPdwgQtpVLKTiEsl170/eCvf/zFzgOxEAwmlvONCIAJuS5CPm92dO15UFrqSSFVs+8VQzAkjWGcmSOmhQIJCTYaoVdEUCogKBUQekWYwIuMIQAhJITlQLkZWG4T7FQW0kmjYjh1GOMcK19mELTtpJUOw4I2/qPPPNbxfTBTLLlJhTC5AJgpXgHe8YU9X7es1GfCoAzWWoNIjpkbWXNlg42GV7iC0SvnUBq8CG9kCDoogXXY8DMVgqSCslKwm1qQap2PzKyFcLIzQULBhH7lG2O+q0lIadkuAq/8xK7cX3waAMVzryuE6wqAmYkIeOi7Tyn/cttuy83c540WNINF3apXGLdgwgAjl09juO84StcuR0STAAkBIgEQTfhRjnEl7hJsIJQFt2UOWhasRFPbYgjLhgmDcYKIdgUZJ5OVfrnYXQyDB3ryH/GZDV0vZphYAMyU6+qivgULpHd5do/tNn2oXBwOCaTq5xmQtAAAhUsncfX0QXjDAwABQlogogpj04JYUMwMowOAAad5FmYuWYvsvJsBAEaHjbZF6Kable+N/rszZ/H2+Rd+rvNdXTyRJkwkgIrBeyD/7L+66WxnebQQEMEa8zUIy4FXuIL+3r0Y7T8brbRUlf/fFkiMpQ7BRiPTtghtK26Hm50FHXjjtgQzAjeTtcrFQvczue0fuZ5hbCiArbnn1Av594UP5Lsfd9Mtf9uQeQBCWRg6fwz9x34D7XsQll2h4B2BmFET+pCWg7YVd6DlppXRlhgDzBy6mWZVHhl64pl856cTnsahHPsg8aUPdPXstFOZp4NyMQRQo/Yc7WkSGOjdi6unXwdJCyK22L8LIBIwRoN1gJlL12H2ituib7MZy1JouxlVLhYe/GG+8+lGcULd7FyORT5PZueXnr2VjNzHQMrokKjqsGPmCRcP/wpD549BOalqgHOdla+3mVzv3+N4YYrsVzWeCKFXQsvClZi7+q44bjTVLcPMQiomUIlFuGXX57YfSXhMsKk6zAA6OrqlCfhfbNfOBOWiJqqLXEBC4OKhFzF0/hik7SIMquonhBhnlBKGta4KXkoJbUxdMiSEmFSDCASGgdZVcUnbxdD5YwCAee3vBZvqf0RErEOj3HQm8ILv5XK8dSzOCnMd3d0inyfjrpEfd9LZ9/heMazz88wQysZA796Y+RQIQHNzFi0tzWhpaYZSapwWMANKycqcluYsiIBUKlV5lkq5AABluZDXGcKyoSy3iqulGVQjhIHelyGUVW+DiKTvFUM7lb3zFPV8PJ8n09HdXeEr0RUCgPu7ftomJA4KIWdrHXIlpo+t/dC5I7h46MXKyjc3Z7H7e48jk0kDAD6b+yp+/X/7kMmkYYyBEALFYhGbN67DN/7x8wAAPwjQef8jePhjf4kPvD9akF/89/P42rd2Yc3dnfC8oGEqIIRAoejh/g+sx5/csRwAcOLkGXzy0c+DiEBE0IGHee3vRctNt0IHXjXUBoyUiozWA7Ks23/w5Y6BWDgsAKCjp0eAiEmYRx0306bDwNQyT1LFru5lCFmVMBGhuTmLpqYMmpoy19WAZE6tBiTP0ulEAxzIBkPZLjQUFs6bjT+9s73yXu+J0yiMjEJKCeYo7ujvfRle4QqEVFU6AaHDwDipTFvoiEdBxB09PSLaAszU09lp7v/anjkE8QnfK1VXvkZJ+nv3QvseIOoTvTCM4vRkNILa/8Mw8kSJDUjsAwEwxkA3GMyMUtnH+7csQ8pRCMLInrz08quQUkZrDAaEgA489PfuraM9XizyvRIT4eH7c3vm9HR2GjCT2Nr1vATAssh/ZacyrWxCXZOmQVg2ChdPYrT/LKRlj/Pxifo1Mn7Xm0d1zyKhOpZAypFI2dWRtiWkYCyek8XdW94FZoalJIYLBRw+chyuY8Mkho8ZUtkY7T+LwsWTUVxSpVewCbWdamol4p0AeGvX81K90LVNd6DbNmweNKHPcREjplrAhB6unjkAEmIarur6LnEsSKnglUZw6qWfVzLECtVCYmi4gI57Pohs2kYQhLAshUOHj+HSpX5kMqmqABDpAgmBwTMH0TRncZx+J3UIEib0mdk8+NB39z351EObQwEizgi6Q9nurUHgV9SfmSGUhZFLZ+AN1++pqYDrOFOeSxSp/9DgAK5drR9DV/sxPDiA1bcsixQ9pmHfK68jCEPUeumIS4aQCuXhAYxcPgOhrKq7BUQQ+Gw57qrShZN3VIygYdyjlMMEmCpRBDYaQ33HgSShmQJw/O6HP/RBWJaCmaLQCIBSCkpZdcMwYe7cOdiwfg0IgGVFhvbV3x6EbVkNYweOEQ5fOA42um57Etgo5YAE3QMAoqO7WzLR3Tr0icH1ln/4CspDl6e8+kSEMAzR0pLF9g//GWbNbEUYjs/YGrwYf5brBhGh7HlYuXwZ2mbPhDYGRISz5/pw4tSbcGr3f50EGEJaKA1dhle4AhJV+hkQOvQBprs7urulSB8SK8C8SodBJciMVlFg9Op5mNCfnIGED0HwfR/Lb16KeXPbsGL5Mvi+X2tVGr83oVwigW7etA4AoGPr/9r+gxgaLkApNcGb0bsm9DF65Xyd/SIQ6TAAE6/KHsFyYQi3WbbrGDaVTIIIYKNRGuyblvoTCFobbNqwBsyMTRvXxIK/saKwMQYp18GWTWsBACIW5N5X9qMS4kwAHDNSGrwYb4MqmYaNsSzXDTXdJhi8iUiAarGRROgV4Y9cgxByysbPGIN0OoXNm9aCiLBp/Rpks5m6PKAhNNAwokibFi6Yj5UrogKIUgrFYgkHDh2F6zqN1b8iAYYQEv7oIEKvCNRE9QRwVJ3CRiGAVQwz5kURFS+Dct2Lk4Hv+7hp4Xwsv3kJAGDJ4oVYuvgm+J5/ff4bCEEIguf5WLfmVriOUwmgjh47gQt9l2BZ1uSulgRCv4ygVIAQom4hGQYgWi0YWBSVoVnUvhiUCmCtp6W9nudjbfstsCwLQRBCCIH161bD88cXLMZLAA01+rYt6wGgstp7X30dnudHDE0GRGCtEZRH6srxzCwinrFIEGG2qdkjCQ2hV0R0TDUNIGBLbLASTFs2rYOU1ye2UZk0DDVmzGjBhnWrAUT5BAN45bUDsBrkHBOQA4ARlkdrKEriDg0CzxYAZSNk9TpoAm/SD4yFGc3NWLd2FQBUVmjN6pWYPbN1isSi8q7n+XjXzUuwYP7cSmZ56VI/eo+fguM606o+mWDsFqSoZA5qFgx2GsX3Y0PSyYCZsWzZYixZtBAA4iQFWDB/LpYvXwrfv44doHqbTkTwgwCbN66NXWFEy/4Db+Dq4BCsOPubMm2mYfUYINhT2EhTgyAIsWlDO1w3CoF7j5+CMQZKKWzeuLaSwTWCsRuA2cCxLdy2OdpOCfEv7/ttQvjbBoJA3viyMoPE1K0/EO3R9fF+1VrjP3/xy4rxW7dmFRw7KSo3oL7mUeT+QsybOwerbl0BIAp/Pd/H6wfegGPbdWWvqQDJ8TaDiACGLwAuRBKunyGtqSczCcKUGxU2RkaK+N+XXsHwcAEA4LoOlJxYoHX7nwie56F99Uo0ZdIV9T9x4gzOnu+Lwt9plt2Fssc8YY6zxGHBjIGopF1PjHIyAKYeBQJRIAQAfZcu4/jJM7hw4SIAQGszCZ6afgkiGOaK+ic49712AMViaWruL4YkB7TcTPKV6DlHhpZBA4KAs9G5XbVUDDawUk0gOfUoMEIczT1x8gyGhoZx4uSbdaRMCFXeobVGtimDTRui8Dcxpvte3Q8l5bTqDFFSJ6HcpvjMIPkOmYhnnBVMdJjqKmAEYwysVBbKdutenCoc7T0JZsaR3hP1HE4A1TJ/5P6WLV2ExYsWgpkhpcDVwWs4cvQEHMeepgAMlJ2ClcpGh611abEAmA8LsHmN2aDOtrKGctKwM62RO5xiNpis1rHek0ilXJw4eQbGmEkDoeQgNDKAATaub4eUohL+Hjh4BAMDV6cW/tbgNEbDzsyActJgrnohBig6ScJrQrDYGwReWZCo1I6iAyCJVOu8SI2m8L2EsKHhAs6eu4BMJo1z5/tw5eq1uv8b0lqDQymB27dsqPt/7yv7o8LpNNwfxYykWufFjRZVUgUJEQTlspa8VxTbTS/AR6SywDXJKxuDzKyFUbPDFKSezDl79gIGrgwi5Tq4dm0Yp8+crft/InKJCEEQoK1tNtpXrwQQZX9aa7y2/xDsaao/xwc5mVk3Relw8hzMUlkgpjf8W3Fc9HR2amL8Uiq7WhIjApsQTnYW3JY5MDqcdBsktB07fgrlsgcpJXzfx7HeU3X/T8A/KM7+Vt2yHK0zWuIVJ5x+8xzOvHkeju3EPRFUNxrjIxgdINUyB052JrhmGxNgpLJhiH/Z09mpBQBojZ+FoUdckzIlwVDLguVT2wbxhGO9J2voIByNf1+vqkRIiim6kkxpHbu/fftx7doQgOhMoTqCio1oSAoDzQuWx+pfmwaTCEMPEvRvQHww4gv+TeiXj1iWTRxrQVJSapqzBG7z7Em1QMb++fiJ07CUgtYGlmXh5KkpGMLY86TTKWzemFR/ovmHj53GzNltaJ7RipbWmTVjFppntI4XLBGMDuE2z0bTnCUwYVB3RGZZNgVe+Q3n/JWXwUxqa9fzsiff6e/8h2e/L5T9VQo8XUme4zPB1qVr0Pf68xByYo9ORBgcHMK5C31RNdhEArjQdwmX+69g3ty2CfkXRPB8HytXLMO7bl5cOU4rFD0MOkux4q4FkGPOJZIF6v31zxD65Up5PDlhal2yFkLZUSZYUX82QtmKfP8HTz318WDr/JVKvNC1TQMgrWmXXxwZJKGqOkMEE/jIzr0ZmbZF0DXIEqit4p45ex5Xrw5VzgiVEhgaLuDUmbPjjs+45t2k/LV+zeqomBIftx06eRl9g0WELFAKGOWaUQoY/tgdQAQd+Mi0LUJ23rI65gEYEkr6pZGryjJPA6AXurZpASLu6O4WP8zfe5mZv2M7qco2QIVUoG3FbZC2A5j6LgylVMUgHT9xGp7nVdSXSCAIQvT2nqrMSSq5Mu4lICJIKUEk8Efv3gQAsGKcrxztA4EgxcSjeopFgDGQtoO2FbeP0zJmZttJEbP5zg8+19nf0d0tQMQKAHo6OgyYSX255wmvNPpRqazZWofRCTERWEceoW3l7bh48MVKwMPMGB4uVIqeBw8fjfdboqwMKQQOvXEMIyOjYABBEIAZKJVKGBmJKjVDwwXMntWKJYtvqswrlgMc7D0HRRqh7413gUSgmp5DAqB1gHmr/hhOdmZU0Kk9HleW8Mojl8Mm9wkwU09i6xIESf/Mjq5nH7bTTf/slQr1LXGxPeg/+hKuntoPYadAYKTTqYqRKRZLleSlFoQQSKdTFTyjxSJs24FlRejDMITvB8ikUxXRaW1Q9ic7VOGoQYoEtF/CrGXr0XbLu+uYj+eFdiqrvFLh4WdyHd+p7RWqnUW5XNQocYr2vGC5qTuDcnFcNygJiYuHfoWh80chbRe6xhUJISd0FGNbZEyDFplQ6zqCIsPHMZl1JjD+TSAhof2oTyhqkRlTeGHWlpuWgVf8n2Xm4FagC7U9QrW+iQEgnydDjI/pwB8lqYjrnCiDjcG81XeiZeEt0H4ZyrLiMz113VgpmVNrA5LfiYu0auYopSI1p8RWiJoRMU5CVJlffVfcXVpLLjNJRTrwRzVZH23UTl/nnJP+mV357Ud04D+ibEcQUVWkRAAMGIx57Xdh1rL10EE5VvvJIsWJvUBC89izwYmASEQ3TwIPs5atj1YeDGBMxkekle2IMAw++aPH7jna0d0ta1cfE1GdNBXu6Or5hpNp/vQfbKPkyjvQsjBplEy2SoWEwM1kLW906PHdXZ2fmXKjZPI8aZXdkd/zEyfd9JE/rFbZxWhbcds71yobY6k2S1+a3WOnJmqW5grDI5dO4eqZAygPvd3N0iHADLd5FlqXrkV27jvfLJ0gi9rlH3pKlRe27XbS02iXv3AcpaGkXT42WtNql49ulgllw21pG9Mu7ydn+LW0xu3yzdIvFrqd8wP3P/XUJ4Ibb5evYq7cvtiR3/N1y5nuhYnzKA1ehD96DaFfis4br1MjJKmg7BTszAykWudN/8KEX3p812P3fgZ4Gy5M1HyhcmXmga6endKy/0lO88qM9orwSwUE5RGE5dGpX5kxOho86ZWZYTbB3+x6bPvTeFuvzNTMrRjGz/e0C0d9U9nu+8LAg9bB7+/SlOUiDLznw6D8qR/m7zv4Dl2aqkLdZckv7HkEJD5r26kFv5drc37pgtH6K7tz278NgG/k2ty0zwZ7Ojt1LpcTANOuv7/324KCDaFf+iID/XaqSSrLFlH0yCFHNfWazhOKo7vGI/m/lucIB4fMzMqyhZNqkszo973SV8LAbNyd2/4tgJHL5cSN3CB9S8eMtRK/70s/nutodwcYO4WS7cpy3uLVWURXZy0LUloIAx9GhwdB9AzCYPeufOfFsTTcCLz1c9Zxl6dz6k21/t0afA9BbGOjVyvLSRFFiQ0bHYXOXJ8MgUR030BEpoSNQRh4JSJ5GODnBOM/Tpr9v05up/9BXJ6uA2baOub6PJhpZ9eeWxi8BUpugNZrQLQQjDYGmkHR9XkwPAKGQegHcA6CDiE0vyXQvl1d9x6tZXJr7jn1Qtc2/VYZT+D/AfLfgRqo+IPUAAAAAElFTkSuQmCC">
<style>
/*__THEME__*/
/* ---- wizstudio layout ---- */
header { align-items:center; flex-wrap:wrap; }
header .spacer { flex:1; }
header .pill { font-size:12px; padding:3px 8px; border-radius:99px; border:1px solid var(--line); background:var(--row-bg); }
header select { padding:3px 6px; font-size:12px; }
nav.tabs { display:flex; gap:4px; padding:0 18px; background:var(--panel); border-bottom:1px solid var(--line); flex-wrap:wrap; }
nav.tabs button { border:0; border-bottom:2px solid transparent; border-radius:0; background:none; padding:10px 12px; color:var(--muted); font-weight:600; }
nav.tabs button.active { color:var(--accent); border-bottom-color:var(--accent); }
section.tab { display:none; }
section.tab.active { display:grid; grid-template-columns:minmax(0,1fr) minmax(0,1fr); gap:16px; padding:16px; align-items:start; }
@media (max-width: 980px){ section.tab.active { grid-template-columns:1fr; } }
.out { width:100%; min-height:380px; font-family:var(--mono); font-size:12.5px; white-space:pre-wrap; }
.status { font-size:12px; color:var(--muted); min-height:16px; }
.status.err { color:var(--err); }
.drop { border:2px dashed var(--line); border-radius:var(--radius); padding:18px; text-align:center; color:var(--muted); cursor:pointer; background:var(--row-bg); }
.drop.over { border-color:var(--accent); color:var(--accent); }
.thumbs { display:flex; flex-wrap:wrap; gap:8px; margin-top:8px; }
.thumb { position:relative; }
.thumb img, .thumbs img { height:72px; border-radius:6px; border:1px solid var(--line); display:block; }
.thumb-x { position:absolute; top:-6px; right:-6px; width:18px; height:18px; padding:0; border-radius:99px; line-height:1; }
label.inline { flex-direction:row; align-items:center; gap:6px; }
.out-tools { display:flex; gap:6px; align-items:center; flex-wrap:wrap; margin-bottom:8px; }
.count { font-size:11.5px; color:var(--muted); font-variant-numeric:tabular-nums; }
.mdview { min-height:320px; max-height:62vh; overflow:auto; padding:10px 12px; border:1px solid var(--line); border-radius:var(--radius-sm); background:var(--input-bg); }
.mdview h1, .mdview h2, .mdview h3 { font-size:14px; margin:10px 0 4px; }
.mdview p { margin:0 0 8px; }
.mdview ul { margin:0 0 8px 18px; }
.mdview pre { white-space:pre-wrap; }
.diff del { background:var(--err-bg); color:var(--err); text-decoration:line-through; }
.diff ins { background:var(--ok-bg); color:var(--ok); text-decoration:none; }
.msg .tools { display:flex; gap:4px; align-items:center; margin-top:6px; }
.msg .tools button { font-size:11px; padding:1px 6px; }
.swipe { font-size:11px; color:var(--muted); }
.listy { display:flex; flex-direction:column; gap:6px; max-height:56vh; overflow:auto; }
.listy button { width:100%; text-align:left; }
.listy button.active { border-color:var(--accent); color:var(--accent); }
.cap-row { display:grid; grid-template-columns:72px minmax(0,1fr); gap:8px; align-items:start; padding:8px 0; border-bottom:1px solid var(--line); }
.cap-row img { width:72px; height:72px; object-fit:cover; border-radius:6px; border:1px solid var(--line); }
.cap-row textarea { min-height:72px; width:100%; }
.sendto select { min-width:0; }
.modal.wide { width:min(760px, calc(100vw - 32px)); }
.lib-hit { flex:1; min-width:0; cursor:pointer; }
.lib-hit b { display:block; }
.krow { display:flex; justify-content:space-between; gap:12px; align-items:center; padding:5px 0; border-bottom:1px solid var(--line); font-size:12.5px; }
.krow span:last-child { display:flex; gap:4px; }
#cap-list { min-height:280px; }
#toast { position:fixed; bottom:16px; right:16px; max-width:min(360px, calc(100vw - 32px)); background:var(--panel); color:var(--ink); border:1px solid var(--line); border-radius:var(--radius); padding:8px 12px; box-shadow:var(--shadow); display:none; z-index:30; }
#toast.show { display:block; }
details summary { cursor:pointer; font-size:12px; color:var(--muted); margin-top:4px; }
details textarea { width:100%; min-height:90px; margin-top:6px; font-family:var(--mono); font-size:12px; }
textarea.big { width:100%; min-height:160px; }
.assist { display:flex; gap:8px; flex-wrap:wrap; align-items:center; margin:8px 0 4px; }
.assist .hint { margin:0; }
.modelbar { display:flex; gap:8px; flex-wrap:wrap; align-items:flex-end; margin-bottom:10px; }
.modelbar label.inline { flex-direction:row; align-items:center; gap:4px; }
.modelbar select { min-width:220px; }
.chatlog { height:56vh; overflow:auto; display:flex; flex-direction:column; gap:8px; padding:4px; }
.msg { padding:8px 11px; border-radius:var(--radius); max-width:88%; border:1px solid var(--line); line-height:1.45; }
.msg.user { align-self:flex-end; background:var(--focus); }
.msg.assistant { align-self:flex-start; background:var(--row-bg); }
.msg .who { font-size:11px; color:var(--muted); font-weight:600; display:block; margin-bottom:2px; }
.msg .body { white-space:pre-wrap; word-break:break-word; }
.msg .body em { font-style:italic; color:var(--muted); }
.msg.rp-empty { max-width:100%; align-self:stretch; background:transparent; border-style:dashed; }
.msg.rp-continue { align-self:center; max-width:60%; opacity:0.75; font-size:12px; padding:4px 10px; }
.msg.rp-continue .body { font-style:italic; color:var(--muted); }
.chips { display:flex; flex-wrap:wrap; gap:6px; }
.chips button { font-size:12px; padding:3px 10px; border-radius:99px; }
.chips button.active { background:var(--accent); color:var(--accent-ink); border-color:var(--accent); }
.char-list { display:flex; flex-direction:column; gap:6px; max-height:34vh; overflow:auto; margin-bottom:8px; }
.char-list button { width:100%; text-align:left; display:flex; flex-direction:column; gap:2px; padding:8px 10px; }
.char-list button.active { border-color:var(--accent); color:var(--accent); }
.char-list .cname { font-weight:600; }
.char-list .ctags { font-size:11px; color:var(--muted); font-weight:400; }
#rp-composer { display:flex; flex-direction:column; gap:6px; margin-top:8px; }
#rp-input { width:100%; min-height:72px; resize:vertical; }
#banner { display:none; margin:10px 16px 0; padding:8px 12px; border-radius:var(--radius-sm); }
#banner.show { display:block; }
</style>
</head>
<body>
<header>
  <h1 style="display:flex;align-items:center;gap:10px"><img src="data:image/png;base64,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" alt="Wiz" width="34" height="34" style="border-radius:50%">wizstudio</h1>
  <span>OCR · Translate · Prompts · Roleplay · Rewrite · Chat · Captions · Extract · T2VA · Ref2VA · Soundscape · Orbit · Character</span>
  <div class="spacer"></div>
  <span id="ollama-state" class="pill">Checking Ollama…</span>
  <button id="refresh-models" title="Reload model list">Models</button>
  <button id="open-lib" type="button" title="Library (Ctrl+K)">Library</button>
  <button id="open-keys" type="button" title="Shortcuts (Ctrl+/)">Shortcuts</button>
  <select id="theme-sel" title="Theme">
    <option value="">auto</option><option value="light">light</option><option value="dark">dark</option>
    <option value="nord">nord</option><option value="nord-light">nord-light</option>
  </select>
  <select id="accent-sel" title="Accent">
    <option value="">blue</option><option value="pink">pink</option><option value="green">green</option><option value="purple">purple</option>
  </select>
</header>
<nav class="tabs">
  <button data-tab="ocr" class="active">🔍 OCR</button>
  <button data-tab="translate">🌐 Translate</button>
  <button data-tab="prompt">✨ Prompt Generator</button>
  <button data-tab="roleplay">🎭 Roleplay Chat</button>
  <button data-tab="rewrite">✍️ Rewrite</button>
  <button data-tab="chat">💬 Chat</button>
  <button data-tab="caption">🏷 Caption</button>
  <button data-tab="extract">📷 Extract</button>
  <button data-tab="t2va">🎬 T2VA</button>
  <button data-tab="ref2va">🧩 Ref2VA</button>
  <button data-tab="sound">🎵 Soundscape</button>
  <button data-tab="orbit">🔄 Center Orbit</button>
  <button data-tab="character">🧑 Character</button>
</nav>
<div id="banner" class="err-box"></div>

<!-- OCR -->
<section class="tab active" id="tab-ocr">
  <div class="col">
    <div class="card">
      <h2>Input</h2>
      <div class="modelbar" data-for="ocr"></div>
      <div class="drop" id="ocr-drop">Drop images or a PDF, click to pick, or paste<input type="file" id="ocr-file" accept="image/png,image/jpeg,image/webp,application/pdf" multiple hidden></div>
      <div class="thumbs" id="ocr-thumbs"></div>
      <div class="hint" id="ocr-pdf-hint"></div>
      <hr>
      <div class="chips" id="ocr-presets"></div>
      <label style="margin-top:8px">Instruction <input id="ocr-instr" value="Extract all text."></label>
      <label style="margin-top:8px">Language hint <input id="ocr-lang" placeholder="Optional, e.g. German or Kannada"></label>
      <div class="bar" style="margin-top:10px">
        <button class="primary" id="ocr-run">Run OCR</button>
        <button id="ocr-clear">Clear images</button>
      </div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Output (markdown)</h2><div data-output="ocr"></div></div></div>
</section>

<!-- Translate -->
<section class="tab" id="tab-translate">
  <div class="col">
    <div class="card">
      <h2>Input</h2>
      <div class="modelbar" data-for="translate"></div>
      <div class="bar">
        <label>From <select id="tr-src"></select></label>
        <button id="tr-swap" title="Swap languages" style="align-self:flex-end">⇄</button>
        <label>To <select id="tr-dst"></select></label>
      </div>
      <label style="margin-top:8px">Text <textarea id="tr-text" class="big" placeholder="Type or paste text…"></textarea></label>
      <div class="grid" style="margin-top:8px">
        <label>Tone <select id="tr-tone"><option value="neutral">Neutral</option><option value="formal">Formal</option><option value="casual">Casual</option></select></label>
        <label class="inline" style="align-self:end"><input type="checkbox" id="tr-breaks" checked> Keep line breaks</label>
      </div>
      <label style="margin-top:8px">Glossary <textarea id="tr-gloss" placeholder="One pair per line: source = target"></textarea></label>
      <div class="drop" id="tr-drop" style="margin-top:8px">Or drop, pick, or paste one image<input type="file" id="tr-file" accept="image/png,image/jpeg,image/webp" hidden></div>
      <div class="thumbs" id="tr-thumbs"></div>
      <div class="bar" style="margin-top:10px">
        <button class="primary" id="tr-run">Translate</button>
        <button id="tr-clear">Clear image</button>
      </div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Translation</h2><div data-output="translate"></div></div></div>
</section>

<!-- Prompt generator -->
<section class="tab" id="tab-prompt">
  <div class="col">
    <div class="card">
      <h2>Idea</h2>
      <div class="modelbar" data-for="prompt"></div>
      <div class="chips" id="pg-presets"></div>
      <div class="hint" id="pg-preset-hint" style="margin-top:6px"></div>
      <label style="margin-top:8px">Your idea <textarea id="pg-idea" class="big" placeholder="e.g. Suhana trying on a red saree in a sunlit boutique"></textarea></label>
      <div class="drop" id="pg-drop" style="margin-top:8px">Optional reference image — drop, click, or paste<input type="file" id="pg-file" accept="image/png,image/jpeg,image/webp" hidden></div>
      <div class="thumbs" id="pg-thumbs"></div>
      <div class="grid" style="margin-top:8px">
        <label>Variations <input id="pg-count" type="number" min="1" max="10" value="1"></label>
        <label>Frame <select id="pg-aspect"><option value="">Unspecified</option><option>1:1</option><option>3:2</option><option>2:3</option><option>16:9</option><option>9:16</option></select></label>
        <label>Trigger <input id="pg-trigger" placeholder="suh4na woman"></label>
      </div>
      <label class="inline" style="margin-top:8px"><input type="checkbox" id="pg-lora"> Character LoRA is loaded</label>
      <label style="margin-top:8px">Also include <input id="pg-also" placeholder="Optional extra requirement"></label>
      <div class="bar" style="margin-top:10px">
        <button class="primary" id="pg-run">Generate prompt</button>
        <button id="pg-clear">Clear image</button>
      </div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Prompt</h2><div data-output="prompt"></div></div></div>
</section>

<!-- Roleplay -->
<section class="tab" id="tab-roleplay">
  <div class="col">
    <div class="card">
      <h2>Characters</h2>
      <div class="modelbar" data-for="roleplay"></div>
      <label>Your name <input id="rp-user" value="Roshan"></label>
      <input id="rp-filter" placeholder="Filter by name or tags…" style="margin-top:8px;margin-bottom:8px">
      <div class="char-list" id="rp-list"></div>
      <div class="bar">
        <button type="button" id="rp-new">New</button>
        <button type="button" id="rp-dup" title="Duplicate selected card">Duplicate</button>
        <button type="button" id="rp-starters" title="Add any missing starter characters">Add starters</button>
        <button type="button" id="rp-del">Delete</button>
      </div>
      <details id="rp-card" open style="margin-top:10px">
        <summary>Edit character card</summary>
        <div class="grid" style="margin-top:8px">
          <label>Name <input id="rp-name"></label>
          <label>Tags <input id="rp-tags" placeholder="e.g. Japanese · Soft · Tokyo"></label>
        </div>
        <label style="margin-top:6px">Persona <textarea id="rp-persona" placeholder="Looks, personality, speech style, cultural flavor…"></textarea></label>
        <label style="margin-top:6px">Scenario <textarea id="rp-scenario" placeholder="Where and when the story starts…"></textarea></label>
        <label style="margin-top:6px">Greeting (first message) <textarea id="rp-greeting"></textarea></label>
        <label style="margin-top:6px">Example dialogue <textarea id="rp-examples" placeholder="{{user}}: …&#10;{{char}}: …"></textarea></label>
        <label style="margin-top:6px">Notes <textarea id="rp-notes" placeholder="Kept on this machine. Not sent to the model."></textarea></label>
        <div class="bar" style="margin-top:8px">
          <button type="button" class="primary" id="rp-save">Save card</button>
          <button type="button" id="rp-export-card">Export card</button>
          <button type="button" id="rp-import-card">Import JSON</button>
          <input type="file" id="rp-import-file" accept="application/json,.json" hidden>
        </div>
        <div class="hint">Supports {{char}}, {{user}}, {{persona}}, {{scenario}} and {{examples}}. Import accepts SillyTavern cards.</div>
      </details>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>Chat <span id="rp-chat-title" class="hint" style="font-weight:400;margin-left:6px"></span></h2>
      <div class="chatlog" id="rp-log"></div>
      <div class="thumbs" id="rp-thumbs"></div>
      <div id="rp-composer">
        <textarea id="rp-input" placeholder="Your message… (Enter to send, Shift+Enter for a new line, paste an image)"></textarea>
        <div class="bar">
          <button type="button" class="primary" id="rp-send">Send</button>
          <button type="button" id="rp-continue" title="Ask the character to keep going">Continue</button>
          <button type="button" id="rp-stop">Stop</button>
          <button type="button" id="rp-regen">Regenerate</button>
          <button type="button" id="rp-edit">Edit last</button>
          <button type="button" id="rp-img">Attach image</button>
          <input type="file" id="rp-file" accept="image/png,image/jpeg,image/webp" hidden>
          <button type="button" id="rp-reset">Restart</button>
          <button type="button" id="rp-export">Save .md</button>
        </div>
      </div>
      <div class="status" id="rp-status"></div>
    </div>
  </div>
</section>

<!-- T2VA (perchance.org/wiz-t2va) -->
<section class="tab" id="tab-t2va">
  <div class="col">
    <div class="card">
      <h2>MiniMax H3 T2VA</h2>
      <div class="modelbar" data-for="t2va"></div>
      <div class="hint">Writes MiniMax-H3 text-to-video clips as a reel. Next scene continues from the last clip. Same prompt rules as perchance.org/wiz-t2va.</div>
      <div class="grid" style="margin-top:8px">
        <label>Type <select id="t2-type" data-keep></select></label>
        <label>Duration <select id="t2-dur" data-keep><option value="6">6s</option><option value="8">8s</option><option value="10">10s</option><option value="15" selected>15s</option></select></label>
      </div>
      <label style="margin-top:8px">Direction (optional) <textarea id="t2-dir" data-keep placeholder="What happens. Leave empty for a random seed situation in this movie type."></textarea></label>
      <label style="margin-top:8px">Locked characters (one per line: <code>Name: fixed appearance and outfit</code>) <textarea id="t2-bible" data-keep></textarea></label>
      <div class="bar" style="margin-top:6px"><button type="button" id="t2-lock">Lock from scene</button><button type="button" id="t2-bible-clear">Clear</button></div>
      <label style="margin-top:8px">Scene location (empty = AI picks the place) <textarea id="t2-loc" data-keep placeholder="NAME || description"></textarea></label>
      <details style="margin-top:6px"><summary>Location builder</summary>
        <div class="grid" style="margin-top:6px">
          <label>Category <select id="t2-lcat"></select></label>
          <label>Place <select id="t2-lplace"></select></label>
          <label>Side <select id="t2-lside"></select></label>
          <label>Era <select id="t2-lera"></select></label>
          <label>Time <select id="t2-ltime"></select></label>
          <label>Weather <select id="t2-lweather"></select></label>
          <label>Mood <select id="t2-lmood"></select></label>
          <label>Style <select id="t2-lstyle"></select></label>
        </div>
        <label style="margin-top:6px">Location idea (for ✨ Generate) <input id="t2-lidea" placeholder="e.g. a cramped 1980s Bombay video parlour"></label>
        <div class="bar" style="margin-top:6px">
          <button type="button" id="t2-lgen">✨ Generate location</button>
          <button type="button" id="t2-lrand">🎲 Random location</button>
          <button type="button" id="t2-linstant">⚡ Instant (no AI)</button>
          <button type="button" id="t2-lenh">Enhance</button>
          <button type="button" id="t2-lroll">Roll all</button>
          <button type="button" id="t2-lany">All to Any</button>
        </div>
      </details>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="t2-next">Next scene</button>
        <button type="button" id="t2-gen">From input</button>
        <button type="button" id="t2-reset">Reset reel</button>
      </div>
      <div class="hint" id="t2-reel" style="margin-top:6px"></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Clip</h2><div data-output="t2va"></div></div></div>
</section>

<!-- Ref2VA (perchance.org/wiz-ref2va) -->
<section class="tab" id="tab-ref2va">
  <div class="col">
    <div class="card">
      <h2>MiniMax H3 Ref2VA</h2>
      <div class="modelbar" data-for="ref2va"></div>
      <div class="hint">Reference-to-video prompt with subject definitions for your reference images. Up to 9 images (people + places) and 3 voice references.</div>
      <div class="grid" style="margin-top:8px">
        <label>Duration <select id="r2-dur" data-keep><option value="6">6s</option><option value="8">8s</option><option value="10">10s</option><option value="15" selected>15s</option></select></label>
        <label>Dialogue <select id="r2-lang" data-keep><option value="English" selected>English</option><option value="Chinese">Chinese</option><option value="Japanese">Japanese</option><option value="Korean">Korean</option><option value="Spanish">Spanish</option><option value="French">French</option><option value="German">German</option><option value="Italian">Italian</option><option value="Portuguese">Portuguese</option><option value="Russian">Russian</option><option value="Arabic">Arabic</option><option value="Hindi">Hindi (house)</option><option value="none">no dialogue</option></select></label>
      </div>
      <label style="margin-top:8px">People — one per line: <code>Name | woman/man/person | note</code> <textarea id="r2-people" data-keep placeholder="Asha | woman | red saree&#10;Vikram | man |"></textarea></label>
      <label style="margin-top:8px">Places — one per line <textarea id="r2-places" data-keep placeholder="the pink house balcony"></textarea></label>
      <label style="margin-top:8px">Voice references — one per line: <code>Label | person number | timbre</code> <textarea id="r2-audios" data-keep placeholder="Asha voice | 1 | warm low alto, soft Bombay accent"></textarea></label>
      <div class="hint" id="r2-budget"></div>
      <label style="margin-top:8px">Interaction <textarea id="r2-text" data-keep class="big" placeholder="What the people do and say."></textarea></label>
      <div class="bar" style="margin-top:10px"><button type="button" class="primary" id="r2-run">Generate</button><button type="button" id="r2-fill">Fill-in (no AI)</button></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Ref2VA prompt</h2><div data-output="ref2va"></div></div></div>
</section>

<!-- Soundscape (perchance.org/wiz-soundscape) -->
<section class="tab" id="tab-sound">
  <div class="col">
    <div class="card">
      <h2>Soundscape &amp; Music</h2>
      <div class="modelbar" data-for="sound"></div>
      <div class="hint">Writes the MiniMax-H3 overall_soundscape and non_diegetic_music fields for a scene. Same options as perchance.org/wiz-soundscape.</div>
      <div class="grid" style="margin-top:8px">
        <label>Music <select id="sd-music" data-keep></select></label>
        <label>Clip (seconds) <input id="sd-dur" data-keep type="number" min="5" max="600" step="1" value="10"></label>
        <label>Genre <select id="sd-genre" data-keep></select></label>
        <label>Tempo <select id="sd-tempo" data-keep></select></label>
        <label>Era <select id="sd-era" data-keep></select></label>
        <label>Vocal <select id="sd-vocal" data-keep></select></label>
        <label>Lyrics language <select id="sd-lang" data-keep></select></label>
        <label>Soundscape <select id="sd-sound" data-keep></select></label>
      </div>
      <div class="chips" id="sd-durs" style="margin-top:6px"><button type="button" data-dur="10">10s</button><button type="button" data-dur="15">15s</button><button type="button" data-dur="30">30s</button><button type="button" data-dur="60">1m</button><button type="button" data-dur="180">3m</button><button type="button" data-dur="600">10m</button></div>
      <label style="margin-top:8px">Scene <textarea id="sd-text" data-keep class="big" placeholder="Night. A wet Bombay street. She steps out of a cab in a chiffon sari and looks up at the pink house. He watches from the balcony."></textarea></label>
      <div class="bar" style="margin-top:10px"><button type="button" class="primary" id="sd-run">Generate</button><button type="button" id="sd-fill">Fill-in (no AI)</button><button type="button" id="sd-example">Example</button></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Sound</h2><div data-output="sound"></div></div></div>
</section>

<!-- Center Orbit (perchance.org/wiz-center-orbit) -->
<section class="tab" id="tab-orbit">
  <div class="col">
    <div class="card">
      <h2>Center Orbit — 360 T2VA</h2>
      <div class="modelbar" data-for="orbit"></div>
      <div class="hint">Name every wall, corner and the middle, or describe the room / drop a photo and let AI fill the form. Generate sweep and Six angles are built locally, no AI.</div>
      <label style="margin-top:8px">Describe room in words → fill form <textarea id="or-desc" data-keep placeholder="small living room with a TV and a grey fabric couch, wooden floor, window on one side"></textarea></label>
      <div class="drop" id="or-drop" style="margin-top:8px">Source photo → autofill — drop, click, or paste<input type="file" id="or-file" accept="image/png,image/jpeg,image/webp" hidden></div>
      <div class="thumbs" id="or-thumbs"></div>
      <div class="bar" style="margin-top:6px"><button type="button" id="or-interpret">Interpret description</button><button type="button" id="or-autofill">Autofill from photo</button><button type="button" id="or-clear">Clear form</button></div>
      <div class="grid" style="margin-top:8px">
        <label>Style <input id="or-style" data-keep placeholder="Live-action, 1986 Bombay, Eastman Color, fine film grain"></label>
        <label>Clip <select id="or-dur" data-keep><option value="6">6s</option><option value="10" selected>10s</option><option value="15">15s</option></select></label>
        <label>Middle sits <select id="or-where" data-keep><option value="above" selected>Hanging above the camera</option><option value="floor">On the floor around the camera</option><option value="waist">Waist height around the camera</option></select></label>
      </div>
      <label style="margin-top:6px">Place <input id="or-place" data-keep placeholder="A small creek hut on the Gorai shore, about 3 by 3 metres."></label>
      <label style="margin-top:6px">Light <input id="or-light" data-keep placeholder="Kerosene light from the hurricane lamp only"></label>
      <label style="margin-top:6px">Soundscape <input id="or-sound" data-keep placeholder="A faint kerosene hiss, creek water, mangrove insects. No voices."></label>
      <div class="grid" style="margin-top:6px">
        <label>North wall <input id="or-north" data-keep></label><label>East wall <input id="or-east" data-keep></label>
        <label>South wall <input id="or-south" data-keep></label><label>West wall <input id="or-west" data-keep></label>
        <label>Northeast corner <input id="or-ne" data-keep placeholder="leave empty if bare"></label><label>Northwest corner <input id="or-nw" data-keep placeholder="leave empty if bare"></label>
        <label>Southeast corner <input id="or-se" data-keep placeholder="leave empty if bare"></label><label>Southwest corner <input id="or-sw" data-keep placeholder="leave empty if bare"></label>
        <label>Middle <input id="or-middle" data-keep></label>
      </div>
      <div class="bar" style="margin-top:10px"><button type="button" class="primary" id="or-run">Generate sweep</button><button type="button" id="or-six">Six angles</button><button type="button" id="or-template">Load template</button></div>
      <div class="hint" id="or-notes" style="margin-top:6px"></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Clips</h2><div data-output="orbit"></div></div></div>
</section>

<!-- Character (perchance.org/wiz-character) -->
<section class="tab" id="tab-character">
  <div class="col">
    <div class="card">
      <h2>Character Generator</h2>
      <div class="modelbar" data-for="character"></div>
      <div class="hint">Empty fields are filled by Develop. Your text is never replaced. Outputs feed a T2VA locked character and a Krea2 / Qwen turnaround sheet.</div>
      <div class="grid" style="margin-top:8px">
        <label>Name <input id="cg-name" data-keep></label>
        <label>Role / archetype <input id="cg-role" data-keep></label>
        <label>Genre / type <select id="cg-genre" data-keep></select></label>
      </div>
      <label style="margin-top:8px">Seed idea <textarea id="cg-seed" data-keep placeholder="One line is enough: a night-shift nurse who keeps a notebook of every patient she lost…"></textarea></label>
      <div class="bar" style="margin-top:6px"><button type="button" id="cg-dice">🎲 Seed</button><button type="button" id="cg-new">New</button></div>
      <div class="grid" id="cg-fields" style="margin-top:8px"></div>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="cg-develop">Develop</button>
        <button type="button" id="cg-deepen">Deepen</button>
        <button type="button" id="cg-build">Build outputs</button>
        <select id="cg-regen-key"></select><button type="button" id="cg-regen">Regenerate</button>
      </div>
      <label class="inline" style="margin-top:8px"><input type="checkbox" id="cg-ref" data-keep> Sheet from reference image</label>
      <label style="margin-top:6px">Visual line (T2VA character lock) <textarea id="cg-visual" data-keep></textarea></label>
      <label style="margin-top:6px">Turnaround sheet prompt (Krea2 / Qwen Image Edit 2.1) <textarea id="cg-sheet" data-keep class="big"></textarea></label>
      <label style="margin-top:6px">Voice description <textarea id="cg-voice" data-keep></textarea></label>
      <div class="bar" style="margin-top:6px"><button type="button" id="cg-to-t2va">Lock in T2VA</button></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>AI output</h2><div data-output="character"></div></div></div>
</section>

<!-- Rewrite -->
<section class="tab" id="tab-rewrite">
  <div class="col">
    <div class="card">
      <h2>Text</h2>
      <div class="modelbar" data-for="rewrite"></div>
      <div class="grid">
        <label>Mode <select id="rw-mode"></select></label>
        <label>Strength / detail <select id="rw-strength"></select></label>
      </div>
      <div class="hint" id="rw-preset-hint" style="margin-top:6px"></div>
      <label style="margin-top:8px">Custom instruction (optional — overrides the mode) <textarea id="rw-extra" placeholder="e.g. Rewrite like a 1940s noir narrator. Keep it under 100 words."></textarea></label>
      <label style="margin-top:8px">Text <textarea id="rw-text" class="big"></textarea></label>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="rw-run">Rewrite</button>
        <button type="button" id="rw-diff">Show changes</button>
        <button type="button" id="rw-use">Use result as source</button>
        <button type="button" id="rw-clear">Clear</button>
      </div>
      <div id="rw-diffbox" class="mdview diff" hidden style="min-height:0;margin-top:8px"></div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Result</h2><div data-output="rewrite"></div></div></div>
</section>

<!-- Chat -->
<section class="tab" id="tab-chat">
  <div class="col">
    <div class="card">
      <h2>Threads</h2>
      <div class="modelbar" data-for="chat"></div>
      <label>Title <input id="ch-title"></label>
      <div class="bar" style="margin-top:8px">
        <button type="button" class="primary" id="ch-new">New thread</button>
        <button type="button" id="ch-del">Delete</button>
      </div>
      <input id="ch-filter" placeholder="Filter threads…" style="margin-top:8px">
      <div class="listy" id="ch-list" style="margin-top:8px"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>Chat</h2>
      <details><summary>System prompt for this thread</summary>
        <textarea id="ch-sys" placeholder="Optional. Falls back to Advanced when empty."></textarea>
      </details>
      <div class="chatlog" id="ch-log"></div>
      <div class="thumbs" id="ch-thumbs"></div>
      <textarea id="ch-input" style="width:100%;margin-top:8px" placeholder="Message… (Ctrl+Enter to send, paste an image)"></textarea>
      <div class="bar" style="margin-top:8px">
        <button type="button" class="primary" id="ch-send">Send</button>
        <button type="button" id="ch-stop">Stop</button>
        <button type="button" id="ch-regen">Regenerate</button>
        <button type="button" id="ch-edit">Edit last</button>
        <button type="button" id="ch-clearimg">Clear image</button>
        <button type="button" id="ch-export">Save .md</button>
        <button type="button" id="ch-lib">To library</button>
      </div>
      <div class="status" id="ch-status"></div>
    </div>
  </div>
</section>

<!-- Caption -->
<section class="tab" id="tab-caption">
  <div class="col">
    <div class="card">
      <h2>Batch</h2>
      <div class="modelbar" data-for="caption"></div>
      <div class="chips" id="cap-presets"></div>
      <div class="hint" id="cap-hint" style="margin-top:6px"></div>
      <label style="margin-top:8px">Trigger word <input id="cap-trigger" placeholder="suh4na woman"></label>
      <div class="drop" id="cap-drop" style="margin-top:8px">Drop images, click to pick a batch, or paste<input type="file" id="cap-file" accept="image/png,image/jpeg,image/webp" multiple hidden></div>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="cap-run">Caption all</button>
        <button type="button" id="cap-stop">Stop</button>
        <button type="button" id="cap-zip">Download .txt zip</button>
        <button type="button" id="cap-clear">Clear</button>
      </div>
      <div class="status" id="cap-status"></div>
    </div>
  </div>
  <div class="col">
    <div class="card">
      <h2>Captions</h2>
      <div class="listy" id="cap-list"></div>
    </div>
  </div>
</section>

<!-- Extract -->
<section class="tab" id="tab-extract">
  <div class="col">
    <div class="card">
      <h2>Image</h2>
      <div class="modelbar" data-for="extract"></div>
      <div class="chips" id="ex-presets"></div>
      <div class="hint" id="ex-hint" style="margin-top:6px"></div>
      <div class="drop" id="ex-drop" style="margin-top:8px">Drop one image, click to pick, or paste<input type="file" id="ex-file" accept="image/png,image/jpeg,image/webp" hidden></div>
      <div class="thumbs" id="ex-thumbs"></div>
      <label style="margin-top:8px">Who to read <input id="ex-focus" placeholder="Optional. The woman on the left, or everyone"></label>
      <label class="inline" style="margin-top:8px"><input type="checkbox" id="ex-lora" checked> LoRA-safe — leave face, hair, skin and body type out</label>
      <label style="margin-top:8px">Also include <input id="ex-also" placeholder="Optional. Which hand holds the cup"></label>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="ex-run">Extract</button>
        <button type="button" id="ex-clear">Clear image</button>
      </div>
    </div>
  </div>
  <div class="col"><div class="card"><h2>Extract</h2><div data-output="extract"></div></div></div>
</section>

<div class="modal-bg" id="lib-modal">
  <div class="modal wide">
    <div class="bar">
      <strong>Library</strong>
      <input id="lib-q" placeholder="Search titles and text…" style="flex:1">
      <select id="lib-filter">
        <option value="">All tools</option>
        <option value="ocr">OCR</option>
        <option value="translate">Translate</option>
        <option value="prompt">Prompt</option>
        <option value="rewrite">Rewrite</option>
        <option value="caption">Caption</option>
        <option value="extract">Extract</option>
        <option value="chat">Chat</option>
        <option value="roleplay">Roleplay</option>
        <option value="t2va">T2VA</option>
        <option value="ref2va">Ref2VA</option>
        <option value="sound">Soundscape</option>
        <option value="orbit">Center Orbit</option>
        <option value="character">Character</option>
      </select>
      <button type="button" id="lib-close">Close</button>
    </div>
    <ul id="lib-list"></ul>
    <div class="hint">Finished OCR, translate, prompt, rewrite, caption and extract runs are kept here, on this machine.</div>
  </div>
</div>
<div class="modal-bg" id="keys-modal">
  <div class="modal">
    <div class="bar"><strong>Shortcuts</strong><span style="flex:1"></span><button type="button" id="keys-close">Close</button></div>
    <div class="krow"><span>Run the open tool</span><span><kbd>Ctrl</kbd><kbd>Enter</kbd></span></div>
    <div class="krow"><span>Stop</span><span><kbd>Esc</kbd></span></div>
    <div class="krow"><span>Switch tool</span><span><kbd>Alt</kbd><kbd>1</kbd><span class="hint">–</span><kbd>0</kbd></span></div>
    <div class="krow"><span>Library</span><span><kbd>Ctrl</kbd><kbd>K</kbd></span></div>
    <div class="krow"><span>This list</span><span><kbd>Ctrl</kbd><kbd>/</kbd></span></div>
  </div>
</div>
<div id="toast" role="status"></div>

<script>
// ===== Perchance generator logic (converted from perchance.org/wiz-* modelText) =====
// ---- Perchance generators (converted from perchance.org/wiz-*; list syntax -> PN nodes) ----
const PC_RW = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "Text Rewriter";
var modeOrder = PN("modeOrder", ["paraphrase", "formal", "casual", "shorter", "longer", "plain", "friendly", "confident", "humanize", "fix", "image_prompt", "video_prompt", "code_prompt", "music_prompt", "system_prompt", "director", "shotlist", "custom"]);
function modeMeta(name){
  let d = {
    paraphrase: {label: "Paraphrase — same meaning, new words", rule: "Paraphrase: keep the exact meaning and every fact, but put it in new wording."},
    formal: {label: "More formal", rule: "Make it more formal and professional. Avoid slang and contractions."},
    casual: {label: "More casual", rule: "Make it more conversational and relaxed. Contractions and everyday words are fine."},
    shorter: {label: "Shorter", rule: "Condense it. Cut filler and repetition, but keep every fact, name, number, and instruction."},
    longer: {label: "Detailer", expand: true, rule: "Expand the existing text in place. Keep the same events, the same order, the same scope, the same ending point, and the meaning of every sentence. Make it longer only by adding sensory, descriptive, and explanatory detail inside each existing sentence or paragraph. Do not add new events, characters, plot, dialogue, or anything after the original ending. No continuation, no conclusion, no title, no padding, no repeated ideas."},
    plain: {label: "Plain English", rule: "Use simpler words, shorter sentences, and clearer structure. Plain English."},
    friendly: {label: "Friendlier", rule: "Make it warmer and more approachable while keeping the meaning."},
    confident: {label: "More confident", rule: "Make it more direct and assertive. Remove hedging like 'maybe', 'I think', 'just', 'sorry'."},
    humanize: {label: "Humanize — remove AI tells", rule: "Rewrite so it sounds like a person wrote it. Remove AI patterns like 'delve', 'seamless', 'in today's world', 'it's important to note', and stiff list-y phrasing. Vary sentence rhythm."},
    fix: {label: "Fix grammar & typos only", rule: "Change ONLY grammar, spelling, punctuation, and typos. Do not reword, do not change style, do not add or remove anything."},
    image_prompt: {label: "Image prompt — pro text-to-image", prompt: "image", rule: "Image prompt mode: turn the raw idea into one professional text-to-image prompt. Write a single flowing paragraph of concrete descriptive phrases in this order: subject (who or what, age, build, expression, clothing, distinguishing details), pose or action, setting and environment, composition and framing (shot type, camera angle, subject placement, depth of field), lens and camera (e.g. 85mm f/1.8, 24mm wide, macro), lighting (direction, quality, time of day, practical sources), colour palette and mood, medium and style (e.g. 35mm film photograph, studio product shot, 3D render, oil painting, anime key visual), then texture and finish. Best practices: be specific and visual rather than abstract; prefer concrete nouns and precise adjectives over filler like 'beautiful', 'stunning', 'masterpiece', or '8k'; describe what is present instead of what is absent (no negations); front-load the subject and the most important details; keep one clear subject and one coherent style, with no contradictory terms; include any text that must appear in the image in quotes only if the input asks for it."},
    video_prompt: {label: "Video prompt — pro text/image-to-video", prompt: "video", rule: "Video prompt mode: turn the raw idea into one professional text-to-video prompt that also works for image-to-video. Write present-tense flowing prose in time order, with these elements: subject and setting established in the first sentence; the action as a short sequence of clear beats (one main action per beat, physically plausible motion); camera (shot size, angle, lens, and movement named plainly, e.g. static locked-off, slow push-in, tracking, handheld with subtle shake, orbit, crane up, rack focus); lighting and colour mood; visual style and pacing (e.g. slow motion, naturalistic documentary, high-contrast noir); and ambient sound or dialogue only if the input has it. Best practices: use a single continuous shot unless the input asks for cuts; give exactly one camera movement per shot unless a combined move is requested; never contradict the camera, lighting, or style; show emotion through small physical cues (breath, eyes, hands, posture); describe what happens and what is seen, not what is absent (no negations); do not describe on-screen text unless requested; if the input mentions a reference image, describe only what moves and changes, not what the image already shows."},
    code_prompt: {label: "Code prompt — pro coding-assistant enhancer", prompt: "code", rule: "Code prompt mode: turn the raw request into one precise, professional prompt for an AI coding assistant. Write it as plain text with short labelled sections, only including the ones that apply, in this order: Goal (one or two sentences stating the outcome); Context (language, framework and versions, relevant files, existing behaviour, environment; use only what the input gives or clearly implies, and never invent file names, APIs, or versions); Requirements (a numbered list of concrete, testable behaviours, including inputs, outputs, and edge cases); Constraints (style, performance, security, dependencies, things not to change); Approach (ask for a brief plan first for non-trivial work, and for minimal focused changes rather than rewrites); Acceptance criteria (how to know it works, including tests to write or run); Output format (e.g. a unified diff, complete files, or an explanation of changes). Best practices: lead with the task, not pleasantries; be specific and unambiguous; replace vague words like 'fast', 'clean', or 'better' with measurable criteria when the input allows, otherwise keep them as stated; state assumptions explicitly as 'Assumption:' lines rather than hiding them; if essential information is missing, add a final line 'Ask me before coding if unclear: ...' with the specific open questions; preserve every code snippet, error message, stack trace, identifier, and path from the input exactly and unchanged; do not write the solution code yourself and do not use markdown code fences."},
    music_prompt: {label: "Music prompt — pro text-to-music", prompt: "music", rule: "Music prompt mode: turn the raw idea into one professional text-to-music prompt for an AI music generator. Write a single compact paragraph of comma-separated descriptors, in this order: primary genre and sub-genre or era; mood and emotional arc; tempo (give a BPM or a clear range) and rhythmic feel; key instruments and sound sources, with how they are played or processed; vocals (voice type, delivery, language, and the theme of the lyrics, or 'instrumental' if none); arrangement and structure (e.g. sparse intro, build, drop, bridge, fade-out); production and mixing style (e.g. lo-fi tape warmth, wide stereo synths, dry intimate vocals, live room reverb); and a reference vibe or use case only if the input gives one. Best practices: be specific rather than generic ('warm fingerpicked nylon guitar' beats 'nice guitar'); keep one coherent genre direction without contradictory tags; never name real living artists or copyrighted songs, describe their sound with genre and production terms instead; keep any lyrics the user supplied exactly as written and do not invent new lyrics unless the input asks for them."},
    system_prompt: {label: "System prompt builder — for AI assistants", prompt: "system", rule: "System prompt builder mode: turn the raw idea into a complete, professional system prompt for an AI assistant, written in the second person ('You are ...'). Use plain text with short labelled sections, including only the ones that apply, in this order: Role (who the assistant is and who it serves); Goal (what success looks like); Context (domain facts, audience, and tools or data it can use, taken only from the input); Behaviour and rules (specific, testable do and don't statements, in order of importance); Tone and style (voice, length, formatting, language); Workflow (how it should handle a typical request, when to ask a clarifying question, and when to say it does not know); Boundaries (what it must refuse or hand off, and how to decline politely); Output format (exact structure, with a short example only if the input needs one). Best practices: be concrete and unambiguous instead of vague adjectives; prefer positive instructions and say what to do instead of only what to avoid; give the reason behind important rules in a few words; avoid contradictory instructions; never invent facts, products, policies, or tools that the input does not mention, and mark gaps as 'Assumption:' lines; keep it as short as the task allows."},
    director: {label: "Director — cinematic scene breakdown", cinematic: true, rule: "Director mode: roleplay as an experienced film director describing how you will shoot this scene. Turn the input into a cinematic scene description in flowing present-tense prose, moving shot by shot. For each beat cover: the action and blocking, facial expressions and emotions (shown through behaviour, not just named), shot type and framing (wide, medium, close-up, extreme close-up, over-the-shoulder, POV, insert), lens choice (e.g. 24mm, 35mm, 50mm, 85mm, macro), camera movement and shooting mode (handheld, Steadicam, dolly in/out, tracking, crane, pan, tilt, whip pan, rack focus, slow motion, locked-off), lighting and colour mood, and pacing/cut rhythm. Keep every story event, character, and line of dialogue from the input, in order, quoting dialogue exactly."},
    shotlist: {label: "Shot list — numbered director's shots", cinematic: true, rule: "Shot list mode: roleplay as a film director and break the scene into a numbered shot list. One shot per numbered entry, in story order, each formatted as: 'SHOT n — [shot type / framing], [lens] | Camera: [movement and shooting mode, e.g. handheld, Steadicam, dolly, crane, slow motion] | Action: [what happens, blocking] | Expression/Emotion: [faces and feelings] | Dialogue: [exact line, or —]'. Add a short lighting/pacing note to a shot only when it matters. Keep every story event and line of dialogue from the input, quoting dialogue exactly."},
    custom: {label: "Custom instruction", rule: "Follow the custom instruction exactly."}
  };
  return d[name] || d.paraphrase;

}
var strengthOrder = PN("strengthOrder", ["light", "medium", "heavy"]);
function strengthMeta(name){
  let d = {
    light: {label: "Light — tweak wording", rule: "Light touch: keep sentence structure, tweak wording."},
    medium: {label: "Medium — rephrase sentences", rule: "Medium: rephrase sentences and vary structure."},
    heavy: {label: "Heavy — free rewrite", rule: "Heavy: rewrite freely, but keep every fact and the meaning."}
  };
  return d[name] || d.medium;

}
function cinematicStrength(name){
  let d = {
    light: "Elaboration: light — stay close to the input; 2 to 4 shots, brief cinematic notes per beat.",
    medium: "Elaboration: medium — a full scene treatment; about 4 to 8 shots with clear camera, lens, and emotion detail.",
    heavy: "Elaboration: heavy — lavish director's treatment; as many shots as the scene needs, rich detail on blocking, micro-expressions, camera, lighting, and rhythm."
  };
  return d[name] || d.medium;

}
function promptStrength(name){
  let d = {
    light: "Detail level: light — compact, about 40 to 70 words, only the essentials.",
    medium: "Detail level: medium — balanced, about 80 to 140 words, every element covered briefly.",
    heavy: "Detail level: heavy — rich and exhaustive, about 150 to 250 words, with precise camera, lighting, material, and motion detail."
  };
  return d[name] || d.medium;

}
function codeStrength(name){
  let d = {
    light: "Detail level: light — a tight prompt of about 60 to 120 words: Goal, key Requirements, and Output format.",
    medium: "Detail level: medium — about 150 to 300 words with every applicable section kept brief.",
    heavy: "Detail level: heavy — a thorough specification of about 300 to 550 words with full requirements, edge cases, constraints, and acceptance criteria."
  };
  return d[name] || d.medium;

}
function musicStrength(name){
  let d = {
    light: "Detail level: light — about 20 to 40 words, genre, mood, tempo, and two or three key instruments.",
    medium: "Detail level: medium — about 50 to 90 words covering every element briefly.",
    heavy: "Detail level: heavy — about 100 to 160 words with full arrangement, production, and vocal detail."
  };
  return d[name] || d.medium;

}
function systemStrength(name){
  let d = {
    light: "Detail level: light — a compact system prompt of about 80 to 150 words: Role, Goal, key rules, and Output format.",
    medium: "Detail level: medium — about 200 to 350 words with every applicable section kept brief.",
    heavy: "Detail level: heavy — a thorough system prompt of about 400 to 700 words with full rules, workflow, boundaries, and an output example where useful."
  };
  return d[name] || d.medium;

}
function expandStrength(name){
  let d = {
    light: "Length: about 1.3 times the original — a little added detail per sentence.",
    medium: "Length: about 1.6 times the original — clear added detail in every paragraph.",
    heavy: "Length: about 2 times the original — rich detail throughout, still inside the existing sentences and paragraphs."
  };
  return d[name] || d.medium;

}
function modeOptions(){
  let out = [];
  let keys = ["paraphrase", "formal", "casual", "shorter", "longer", "plain", "friendly", "confident", "humanize", "fix", "image_prompt", "video_prompt", "code_prompt", "music_prompt", "system_prompt", "director", "shotlist", "custom"];
  try {
    if (typeof modeOrder !== "undefined" && modeOrder && modeOrder.selectAll) {
      keys = modeOrder.selectAll.map(function (item) {
        try { return String(item.evaluateItem); } catch (e) { return String(item); }
      });
    }
  } catch (e) {}
  keys.forEach(function (k) {
    let m = modeMeta(k);
    out.push(k + "=" + ((m && m.label) || k));
  });
  return out.join("|||");

}
function strengthOptions(){
  let out = [];
  let keys = ["light", "medium", "heavy"];
  try {
    if (typeof strengthOrder !== "undefined" && strengthOrder && strengthOrder.selectAll) {
      keys = strengthOrder.selectAll.map(function (item) {
        try { return String(item.evaluateItem); } catch (e) { return String(item); }
      });
    }
  } catch (e) {}
  keys.forEach(function (k) {
    let m = strengthMeta(k);
    out.push(k + "=" + ((m && m.label) || k));
  });
  return out.join("|||");

}
function buildRewriter(text, mode, strength, custom){
  let t = String(text || "").trim();
  if (!t) {
    t = "";
  }
  let m = modeMeta(mode);
  let s = strengthMeta(strength);
  let c = String(custom || "").trim();
  let modeRule = (c && mode !== "fix") ? "Custom instruction (this overrides the preset mode): " + c + "." : (m && m.rule ? m.rule : "Paraphrase: keep the meaning, use new wording.");
  let strengthRule = s && s.rule ? s.rule : "Medium: rephrase sentences and vary structure.";
  if (m && m.expand && !c) {
    return "Expand the text below. Output ONLY the expanded text — no preamble, no 'Here is', no 'Sure', no title, no markdown fences, no explanation, no notes, no questions back.\n\nMode: " + m.rule + "\n" + expandStrength(strength) + "\n\nRules that always apply:\n- This is NOT a continuation. Stop exactly where the original text stops. The last sentence of your output covers the same moment as the last sentence of the input.\n- Keep every fact, name, number, date, quote, code block, and URL unchanged.\n- Added detail may only describe, clarify, or explain what is already there; it must not change what happens or add new facts that alter the meaning.\n- Same language as the input. If the input mixes languages, keep that mix.\n- Keep paragraph breaks and the order of ideas.\n- Do not add opinions or disclaimers.\n\nTEXT TO EXPAND (begin):\n" + t + "\n(end of text — do not write past this point)";
  }
  if (m && m.prompt) {
    let kind = m.prompt === "video" ? "video" : (m.prompt === "code" ? "coding" : (m.prompt === "music" ? "music" : (m.prompt === "system" ? "system" : "image")));
    let pRules;
    if (m.prompt === "code") {
      pRules = "\n\nRules that always apply:\n- Keep the user's intent and every explicit detail they gave: languages, frameworks, versions, file names, identifiers, error messages, and code snippets, unchanged.\n- Do not invent facts about the codebase. Where something is unknown, state it as an Assumption or an open question instead.\n- Write the prompt in English unless the input asks for another language. Keep code and identifiers exactly as given.\n- Output one prompt only, as plain text with short section labels. Do not write the solution, do not add alternatives or commentary.";
    } else if (m.prompt === "system") {
      pRules = "\n\nRules that always apply:\n- Keep the user's intent and every explicit detail they gave: the assistant's purpose, audience, tools, rules, and tone.\n- Do not invent facts, products, policies, or tools. Where something is unknown, state it as an Assumption.\n- Write the system prompt in English unless the input asks for another language. Keep names and quoted text exactly as given.\n- Output one system prompt only, as plain text with short section labels. Do not add alternatives, an introduction, or commentary about the prompt.";
    } else if (m.prompt === "music") {
      pRules = "\n\nRules that always apply:\n- Keep the user's idea and every explicit detail they gave: genre, mood, instruments, tempo, vocals, language, and any lyrics (keep supplied lyrics exactly as written).\n- You MAY add specific, plausible musical detail that the idea implies (tempo, arrangement, production), but do not change the core idea.\n- Do not name real living artists or copyrighted songs. Describe their sound with genre and production terms instead.\n- Write the prompt in English unless the input explicitly asks for another language. Keep lyrics in their original language.\n- Output one prompt only: no title, no bullet points, no alternatives, no explanation.";
    } else {
      pRules = "\n\nRules that always apply:\n- Keep the user's subject and every explicit detail they gave: names, counts, colours, places, styles, and any quoted text or dialogue (quote dialogue exactly).\n- You MAY add specific, plausible visual detail that the idea implies (lens, lighting, framing, texture, motion), but do not change the core idea or add new subjects or events.\n- Write the prompt in English unless the input explicitly asks for another language. Keep dialogue and on-screen text in their original language.\n- Output one prompt only: no title, no labels, no bullet points, no alternatives, no explanation.";
    }
    if (c) pRules += "\n- Extra instruction from the user: " + c + ".";
    return "You are an expert prompt engineer for " + (m.prompt === "code" ? "AI coding assistants" : (m.prompt === "system" ? "AI assistant system prompts" : (m.prompt === "music" ? "AI music generation models" : kind + " generation models"))) + ". Rewrite the raw user input below into a high-quality, professional " + kind + " prompt using current best practices. Output ONLY the finished prompt — no preamble, no 'Here is', no 'Sure', no markdown fences, no notes, no questions back.\n\nMode: " + m.rule + "\n" + (m.prompt === "code" ? codeStrength(strength) : (m.prompt === "music" ? musicStrength(strength) : (m.prompt === "system" ? systemStrength(strength) : promptStrength(strength)))) + pRules + "\n\nRAW INPUT:\n" + t;
  }
  if (m && m.cinematic) {
    let cineRules = "\n\nRules that always apply:\n- Keep every story event, character, name, and line of dialogue from the input, in the same order. Quote dialogue exactly.\n- You MAY and SHOULD invent cinematic detail (camera, lens, framing, movement, blocking, expressions, emotion, lighting, pacing), but do not add new plot events or new dialogue.\n- Same language as the input.\n- Do not explain your choices, add disclaimers, or ask questions.";
    if (c) cineRules += "\n- Extra instruction from the user: " + c + ".";
    return "You are a film director. Rewrite the scene below as described. Output ONLY the result — no preamble, no 'Here is', no 'Sure', no markdown fences, no notes.\n\nMode: " + m.rule + "\n" + cinematicStrength(strength) + cineRules + "\n\nSCENE:\n" + t;
  }
  return "Rewrite the text below. Output ONLY the rewritten text — no preamble, no 'Here is', no 'Sure', no markdown fences, no explanation, no notes, no questions back.\n\nMode: " + modeRule + "\nStrength: " + strengthRule + "\n\nRules that always apply:\n- Keep the exact meaning and every fact, name, number, date, quote, code block, and URL unchanged.\n- Same language as the input. If the input mixes languages, keep that mix.\n- Keep paragraph breaks and the order of ideas.\n- Do not add new facts, opinions, or disclaimers.\n\nTEXT:\n" + t;

}
function fallbackRewriter(text){
  String(text || "").trim();
}
return {title, modeOrder, modeMeta, strengthOrder, strengthMeta, cinematicStrength, promptStrength, codeStrength, musicStrength, systemStrength, expandStrength, modeOptions, strengthOptions, buildRewriter, fallbackRewriter};
})();


const PC_T2VA = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "MiniMax H3 T2VA Prompt Generator";
var sparks = PN("sparks", [["hollywoodSilent", ["a crowded 1920s train platform, a woman in a cloche hat searches the crowd", "a factory floor of spinning looms, a foreman points at a jammed wheel", "a mansion ballroom, a dropped glove on the parquet", "a slapstick kitchen, a baker and a cop fight over a pie", "a rooftop chase above a brick city, laundry lines whipping"]], ["hollywoodGolden", ["a 1930s newspaper office at night, a reporter slams a phone", "an ocean-liner ballroom, a singer in sequins holds a last note", "a studio backlot, a director frames a starlet against a fake skyline", "a marble staircase in a penthouse, a couple pause mid-argument", "a nightclub booth, a gangster slides an envelope across the table"]], ["hollywoodNoir", ["rain on a midnight diner window, a detective turns his cup", "a venetian-blind office, a woman in a pale coat closes the door", "a parking garage, a man checks a revolver by a column", "an alley neon puddle, footsteps approach off-screen", "an interrogation room, a single lamp over a metal table"]], ["hollywood1970s", ["a dusty Los Angeles street at magic hour, a cop leans on a sedan", "a cramped kitchen, a couple fight over a stack of bills", "a highway shoulder, heat shimmer over a stalled car", "a bank lobby, a nervous man watches the clock", "a wood-paneled living room, a TV news report plays ignored"]], ["hollywood1980s", ["a neon shopping mall, teenagers cut through a fountain plaza", "a suburban bedroom, synth posters, a girl tapes a mixtape", "a cop car at night, red-blue bounce on wet asphalt", "a high-school hallway after the bell, lockers slam", "a penthouse party, chrome, shoulder pads, city grid in the windows"]], ["hollywood1990s", ["a New York subway car, a stranger offers a Walkman earbud", "a sunlit diner booth, two friends split a map", "a courtroom corridor, a lawyer whispers to a client", "a starter apartment, a blinking answering machine", "a hacker bedroom, CRT glow on a face"]], ["hollywoodNow", ["a phone-lit bedroom at 2am, a face on a video call", "an open-plan office, a standup meeting nobody wants", "a rideshare back seat in rain, city smearing past", "an airport gate, a delayed-board argument", "a rooftop party, portable speaker, skyline haze"]], ["bollywoodClassic", ["a village well at dawn, a woman fills a brass pot", "a black-and-white courthouse verandah, a lawyer in a coat", "a garden swing under a mango tree, two lovers almost speak", "a steam-train platform, a family waves a red cloth", "a haveli courtyard, a letter is read aloud"]], ["bollywood1970s", ["a Bombay chawl corridor, a radio plays a filmi song", "a smoky cabaret, a dancer in sequins holds the room", "a ravine road, a dacoit on horseback blocks a jeep", "a police station veranda, a man in handcuffs waits", "a boxing akhada, sweat and a shouting coach"]], ["bollywood1980s", ["a wet city street at night, Eastman reds, a heroine in a chiffon sari", "a college canteen, two rivals share one bench", "a family living room with a Doordarshan glow", "a palace corridor, a villain in a white suit laughs", "a temple courtyard, a lost locket on stone"]], ["bollywood1990s", ["a college festival stage, a boy grabs a mic", "a Swiss-looking meadow, a heroine spins in a windbreaker", "a haveli night rain song, two faces under one shawl", "a Delhi traffic jam, a hero on a motorcycle", "a mansion party, a secret is mouthed across the room"]], ["bollywoodNow", ["a Mumbai high-rise kitchen, a couple pack tiffin", "a sangeet hall, LED lights, aunts pulling someone to dance", "a startup office at 11pm, two people leftover at desks", "monsoon traffic, a scooter with two riders and a shared raincoat", "a Goa beach shack, a phone recording a confession"]], ["anime", ["a high-school rooftop at sunset, a girl holds an unopened letter", "a rain neon crosswalk, two students share an umbrella", "shrine stairs in summer cicadas, a boy looks back", "an empty classroom after hours, chalk dust in a shaft of light", "a train platform, a last-car wave as doors close"]], ["cartoon", ["a chase through a cartoon house, doors and floors misbehave", "a desert road, a tiny car vs a huge boulder", "a pie-fight kitchen, frosting in the air", "a talking-animal forest, a map that keeps changing", "a city skyline pan, a character runs the rooftops"]], ["commercial", ["an extreme close-up of a soda can cracking open on ice", "a car on wet night asphalt, logo withheld until the last beat", "a perfume portrait, fabric and a slow turn to camera", "a phone sliding out of a box, fingers only", "a kitchen before-and-after on a white shirt stain"]], ["musicVideo", ["a warehouse strobe, a performer walks the camera line", "a rooftop silhouette against a hard sun", "a one-take corridor, backup dancers peel off", "a packed club floor, a chorus hit", "a desert wide, a single figure and a speaker stack"]], ["soap", ["a hospital corridor, two people freeze at a door", "a kitchen accusation over an uneaten breakfast", "a boardroom, a will is slid across glass", "a balcony at night, a almost-kiss interrupted", "a living room, a hidden text lights a phone"]], ["horror", ["a dark hallway, a door at the end is one inch open", "a basement bulb, something moves behind the washer", "woods at blue hour, a flashlight finds a shoe", "a bathroom mirror, a delay in the reflection", "an empty playground at night, one swing moving"]], ["documentary", ["an interview chair against a plain wall, hands twisting", "a factory line, a worker explains a machine", "a protest street, a handheld camera following", "a cook's home kitchen, a family recipe", "a desk of photographs, a voice off-screen names them"]], ["exploitation", ["a grindhouse neon motel, a couple argue in the parking lot", "a nightclub back room, a deal goes bad", "a dusty fight-ring basement, a crowd leans in", "a drive-in lot, a film flicker on faces", "a cheap hotel corridor, someone runs with a bag"]]]);
function fmtStamp(sec){
  let s = Number(sec);
  if (!(s > 0)) {
    s = 0;
  }
  let mm = Math.floor(s / 60);
  let ss = Math.floor(s % 60);
  let ms = Math.round((s - Math.floor(s)) * 1000);
  if (ms === 1000) {
    ss += 1;
    ms = 0;
  }
  if (ss >= 60) {
    mm += 1;
    ss = 0;
  }
  return String(mm).padStart(2, "0") + ":" + String(ss).padStart(2, "0") + "." + String(ms).padStart(3, "0");

}
function clampDur(v){
  let n = parseFloat(v);
  if (isNaN(n) || n < 1) {
    n = 1;
  }
  if (n > 15) {
    n = 15;
  }
  return n;

}
function randomTypeName(){
  let all = sparks.selectAll;
  let i = Math.floor(Math.random() * all.length);
  return all[i].getName;

}
function randomSpark(name){
  let node = sparks[name];
  if (!node) {
    return "two unnamed people in a room, one continuous action";
  }
  return String(node);

}
function typeMeta(name){
  let d = {
    hollywoodSilent: ["Live-action, 1920s silent film, high-contrast black and white, slight flicker", "none", "yes", "score"],
    hollywoodGolden: ["Live-action, 1930s-1940s Hollywood studio picture, glossy lighting, elegant camera", "English", "no", "score"],
    hollywoodNoir: ["Live-action, 1940s-1950s film noir, hard shadows, wet streets, venetian-blind light", "English", "no", "score"],
    hollywood1970s: ["Live-action, 1970s New Hollywood, grain, zoom lenses, naturalistic light", "English", "no", "none"],
    hollywood1980s: ["Live-action, 1980s Hollywood, saturated color, neon, modest camera shake", "English", "no", "score"],
    hollywood1990s: ["Live-action, 1990s Hollywood, clean 35mm, practical light", "English", "no", "none"],
    hollywoodNow: ["Live-action, contemporary cinematic, digital, naturalistic", "English", "no", "none"],
    bollywoodClassic: ["Live-action, 1950s-1960s Hindi cinema, studio light, graceful blocking", "Hindi", "no", "score"],
    bollywood1970s: ["Live-action, 1970s Hindi masala film, bold color, zoom, dramatic blocking", "Hindi", "no", "score"],
    bollywood1980s: ["Live-action, 1980s Bollywood, Eastman color, subtle camera shake", "Hindi", "no", "score"],
    bollywood1990s: ["Live-action, 1990s Hindi romantic drama, glossy travel light, song-ready frames", "Hindi", "no", "score"],
    bollywoodNow: ["Live-action, contemporary Hindi cinema, naturalistic with filmy punch-ins", "Hindi", "no", "none"],
    anime: ["2D-animated, anime, clean linework, dramatic lighting, cinematic camera", "Japanese", "no", "score"],
    cartoon: ["2D-animated, theatrical cartoon, graphic shapes, squash-and-stretch, bright color", "English", "no", "score"],
    commercial: ["Live-action, polished TV commercial, product-hero lighting, exact graphic end frame", "English", "no", "score"],
    musicVideo: ["Live-action, music-video, rhythmic cutting, stylized light", "English", "no", "score"],
    soap: ["Live-action, TV soap, medium close-ups, interior practicals, reaction coverage", "English", "no", "none"],
    horror: ["Live-action, horror, motivated practicals, slow push-ins, negative space", "English", "no", "none"],
    documentary: ["Live-action, observational documentary, handheld, available light", "English", "no", "none"],
    exploitation: ["Live-action, grindhouse/exploitation, cheap neon, grain, blunt blocking", "English", "no", "none"]
  };
  let row = d[name] || ["Live-action, cinematic", "English", "no", "none"];
  return row.join("|||");

}
function t2vaRules(dur, style, language, silent, music){
  dur = clampDur(dur);
  let endTs = fmtStamp(dur);
  let durLabel = String(Math.round(dur * 10) / 10) + "s";
  let talk = silent === "yes" ? "No spoken dialogue. Use intertitles in English double quotes for any words. No <d> blocks." : ("Dialogue language default: " + language + ". Only vocalizing people get (S1), (S2). Same person keeps the same ID. Silent people get no ID. First vocal beat: identity + timbre outside <d>. Inside <d>: only [Language] and the exact words. Voiceover: says in an off-screen voiceover: <d>[Language] ...</d> while lips remain completely closed.");
  let musicRule = "non_diegetic_music: exactly N/A. No music, no score, no songs.";
  return "Output exactly these three fields and nothing else (no intro, no markdown fences):\n\nintegrated_multimodal_description: [Shot 1] ...\n\noverall_soundscape: ...\n\nnon_diegetic_music: ...\n\nThis is MiniMax-H3 T2VA (VIDEO_PROMPT_WRITING_GUIDE_base_en.md). No image-alignment header. No <Picture>, <Image>, or <Subject> labels.\nDuration: " + durLabel + ". No timestamp at or after " + endTs + ".\nStyle: Shot 1 MUST open with: " + style + ", then shot size, then composition.\n[Shot 1] has no timestamp. Later shots: [Shot N] At MM:SS.mmm, the camera cuts to ... Times strictly increase and stay before " + endTs + ".\nDo not force three shots. One continuous action = one shot. A cut must change subject, space, state, viewpoint, or time.\nCamera motion in natural English: type + amplitude + speed when useful (Push In, Pan, Tracking Shot, Static Shot, Shake Slightly, etc.).\nDefault camera: a subtle handheld shake (Shake Slightly, small amplitude, slow) in every shot unless the user asks for a locked-off camera.\n" + talk + "\nOn-screen text in English double quotes, verbatim.\noverall_soundscape: exactly N/A. Dialogue only: the only audio is spoken dialogue. Do not describe ambience, room tone, sound effects, or music anywhere.\n" + musicRule + "\nFollow the user's idea. Any rating is allowed if they wrote it. If they did not name a sex act, do not invent one. Do not refuse, moralize, or add consent lectures. Do not add extra named IP.";

}
function buildT2va(mode, userText, dur, typeName, previousText, sceneNum, spark, bibleText, locationText){
  dur = clampDur(dur);
  let meta = String(typeMeta(typeName) || "").split("|||");
  let style = meta[0] || "Live-action, cinematic";
  let language = meta[1] || "English";
  let silent = meta[2] || "no";
  let music = meta[3] || "none";
  let t = String(userText || "").trim();
  let prev = String(previousText || "").trim();
  let n = parseInt(sceneNum, 10);
  if (isNaN(n) || n < 1) {
    n = 1;
  }
  let rules = t2vaRules(dur, style, language, silent, music);
  let head = "";
  if (mode === "next" && prev) {
    head = "Write clip " + n + " of a sequence. Each clip is a separate T2VA file that starts at 00:00.000. Do not repeat the previous clip. Continue from its ending state.\nKeep the same people, faces, wardrobe, ages, and world unless the user direction changes them.\nKeep speaker IDs stable (same person stays S1 or S2). A new speaker gets the next ID.\nAdvance time by seconds or a short story cut. New action only. No recap.\nIf the user direction conflicts with continuity, follow the user and bridge with one action sentence.\nUser direction for THIS clip (may be empty):\n" + (t || "(none — invent the intelligent next beat)") + "\n\nPREVIOUS CLIP:\n" + prev + "\n\n";
  } else if (mode === "start") {
    if (String(bibleText || "").trim()) {
      head = "Write opening clip 1 in this movie type. This is the first clip of a reel.\nCAST: the LOCKED CHARACTERS listed below are the main cast of this clip. Feature them on screen (all of them unless the user direction says otherwise) and establish them and the place clearly. Do not replace them with other people.\nSeed situation (use it only for the setting and story beat unless the user direction overrides it; any people it mentions are replaced by the locked characters):\n" + String(spark || randomSpark(typeName)) + "\nUser direction (overrides the seed if present):\n" + (t || "(none)") + "\n\n";
    } else {
      head = "Write opening clip 1 in this movie type. Establish people and place clearly. This is the first clip of a reel.\nSeed situation (use it unless the user direction overrides it):\n" + String(spark || randomSpark(typeName)) + "\nUser direction (overrides the seed if present):\n" + (t || "(none)") + "\n\n";
    }
  } else {
    if (!t) {
      t = "(empty — invent one simple unnamed scene in this movie type; no sex; no named IP)";
    }
    head = "Rewrite the user input into one T2VA clip in this movie type. Fold their details in. Do not ignore their action.\nUser input:\n" + t + "\n\n";
    if (String(bibleText || "").trim()) {
      head += "CAST: the LOCKED CHARACTERS listed below are the cast. Use them for the people in this clip (anyone the user input mentions who matches one of them is that character). Only add other people if the user input asks for them.\n\n";
    }
  }
  let bible = String(bibleText || "").trim();
  if (bible) {
    head += "LOCKED CHARACTERS (character bible — binding):\n" + bible + "\nWhenever one of these characters appears, describe them with their locked description copied word for word: same age, build, hair, face, distinguishing features, and the exact same outfit. Do not paraphrase, shorten, recolor, or swap any item. Change a locked description only if the user direction above explicitly asks for that change. Characters not in this list may be introduced and described fully.\n\n";
  }
  let loc = String(locationText || "").trim();
  if (loc) {
    head += "LOCKED LOCATION (binding — this clip takes place here):\n" + loc + "\nThe [Shot 1] text MUST contain the locked location description above copied word for word, as its own sentence right after the style, shot size and composition. Every later shot stays in this same place: keep the architecture or landscape, the fixed props and their placement, materials, colours, lighting, time of day, weather and atmosphere exactly as locked. Do not add, remove, move, recolor or swap any fixed element, do not change the time of day or weather, and do not move the action to another place. A seed situation or previous clip that names a different place is overridden by this location. Change it only if the user direction above explicitly asks for that change.\n\n";
  }
  return head + rules;

}
function locationWrite(idea, typeName, knownNames){
  let meta = String(typeMeta(typeName) || "").split("|||");
  let style = meta[0] || "Live-action, cinematic";
  let known = String(knownNames || "").trim();
  return "Design ONE reusable film location for a video prompt, from the idea below. Output ONLY one line in this exact format, nothing else (no intro, no markdown, no bold, no notes):\nNAME || description\nUse two vertical bars || between the name and the description.\n\nNAME: a short stable place name (2-6 words), e.g. 'the harbour diner'." + (known ? " Do not reuse these names: " + known + "." : "") + "\ndescription: one dense English paragraph, physical and visible only, in this order: setting type and interior or exterior; architecture or landscape; key fixed props and exactly where they sit relative to each other; materials and colours; lighting and time of day; weather and atmosphere; then 'Camera-facing layout:' with what is left, right and behind when facing the main wall or view. No people, no actions, no sounds, no music, no camera moves. Every detail must be specific so it can be reused identically in every shot. Fit this look: " + style + ".\n\nIDEA:\n" + String(idea || "").trim();

}
function locationEnhance(name, desc, typeName){
  let meta = String(typeMeta(typeName) || "").split("|||");
  let style = meta[0] || "Live-action, cinematic";
  return "Improve this film location description so it can be reused identically in every shot of a video prompt. Keep everything it already says (same place, same items, same colours) and add missing specifics in this order: setting type and interior or exterior; architecture or landscape; key fixed props and exactly where they sit; materials and colours; lighting and time of day; weather and atmosphere; then 'Camera-facing layout:' with what is left, right and behind. Physical and visible only: no people, actions, sounds, music, or camera moves. Fit this look: " + style + ".\nOutput ONLY one line in this exact format, nothing else:\nNAME || description\n\nLOCATION:\n" + String(name || "the location").trim() + " || " + String(desc || "").trim();

}
function bibleExtract(clipText, bibleText){
  let clip = String(clipText || "").trim();
  let bible = String(bibleText || "").trim();
  return "Extract a character bible from the video clip prompt below. Output ONLY lines in this exact format, one per on-screen person or recurring creature, nothing else (no intro, no markdown, no bold, no numbering, no code fences, no notes):\nNAME || fixed appearance and outfit\nExample line: the bartender || a broad man in his fifties, ...\nUse two vertical bars || between the name and the description.\n\nNAME: their name, or a short stable label if unnamed (e.g. 'the young woman', 'the bartender').\nfixed appearance and outfit: one dense English sentence with apparent age, build, skin tone, hair (color, length, style), face and distinguishing features, then the full outfit with colors and materials, footwear and accessories. Only physical, visible traits that stay fixed between shots. No actions, emotions, positions, or camera words. If the clip leaves a trait unstated, choose one plausible, specific value that fits the clip and the movie type so it can stay fixed.\n" + (bible ? "These characters are already locked. Do NOT output them again, even if they appear:\n" + bible + "\n" : "") + "If there are no new characters, output exactly: NONE\n\nCLIP:\n" + clip;

}
function fillInT2va(mode, userText, dur, typeName, previousText, bibleText, locationText){
  dur = clampDur(dur);
  let meta = String(typeMeta(typeName) || "").split("|||");
  let style = meta[0] || "Live-action, cinematic";
  let music = meta[3] || "none";
  let t = String(userText || "").trim();
  let body = t;
  if (!body) {
    if (mode === "next") {
      body = "The same people continue from the previous ending into the next physical action.";
    } else {
      body = String(randomSpark(typeName));
    }
  }
  let cast = String(bibleText || "").trim().split("\n").map(l => l.replace(/^-\s*/, "").trim()).filter(Boolean);
  if (cast.length && mode !== "next") {
    body = body.replace(/[.\s]+$/, "") + ". On screen: " + cast.join("; ") + ".";
  }
  let loc = String(locationText || "").trim().replace(/^-\s*/, "");
  if (loc) {
    body = "Setting — " + loc.replace(/[.\s]+$/, "") + ". " + body;
  }
  let shots = "[Shot 1] " + style + ", a medium shot frames this: " + body + " The camera shakes slightly with small amplitude at slow speed, handheld.";
  if (dur >= 8) {
    shots += " [Shot 2] At " + fmtStamp(dur / 2) + ", the camera cuts to a closer view of the same scene continuing. The camera pushes in with small amplitude at slow speed, with a subtle handheld shake.";
  }
  let sound = "N/A";
  let score = "N/A";
  return "integrated_multimodal_description: " + shots + "\n\noverall_soundscape: " + sound + "\n\nnon_diegetic_music: " + score;
}
var locPlace = PN("locPlace", [["urban", ["a rain-slick downtown crosswalk", "a narrow alley behind a noodle shop", "a rooftop bar above a busy avenue", "a 24-hour laundromat", "a multi-storey parking garage", "a subway platform", "a corner bodega", "a hotel lobby"]], ["rural", ["a dusty crossroads gas station", "a barn loft full of hay bales", "a farmhouse porch", "a village tea stall by the road", "a small-town diner", "a grain silo yard", "a roadside motel"]], ["nature", ["a pine-forest clearing", "a mountain lake shore", "a desert canyon floor", "a bamboo grove", "a waterfall pool", "a wildflower meadow", "a mangrove creek"]], ["home", ["a cramped studio apartment", "a family kitchen", "a teenager's bedroom", "a suburban living room", "an attic storage room", "a bathroom with a clawfoot tub", "a balcony with potted plants"]], ["workplace", ["an open-plan office at night", "a hospital corridor", "a police interrogation room", "a restaurant kitchen during service", "a newsroom", "an auto-repair garage", "a hair salon"]], ["fantasy", ["a moss-covered ruined temple", "a wizard's tower study", "a tavern in a mountain pass", "a crystal cavern", "a floating market on river barges", "a royal throne room"]], ["scifi", ["a starship bridge", "a neon megacity street market", "an orbital station observation deck", "a sterile cloning lab", "a Martian greenhouse dome", "a cyberpunk ramen bar"]], ["historical", ["a Victorian gentlemen's club", "a 1920s speakeasy", "a medieval castle kitchen", "a Mughal palace courtyard", "a Wild West saloon", "a 1950s drive-in"]], ["coastal", ["a fishing-harbour pier", "a beach shack bar", "a lighthouse lantern room", "a seaside boardwalk arcade", "a cliffside villa terrace"]], ["underground", ["a basement boxing gym", "an abandoned metro tunnel", "a wine cellar", "a bunker control room", "an underground nightclub"]]]);
var locEra = PN("locEra", ["present day", "near future", "far future", "1920s", "1950s", "1970s", "1980s", "1990s", "Victorian era", "medieval", "timeless fantasy"]);
var locSide = PN("locSide", ["interior", "exterior"]);
var locTime = PN("locTime", ["dawn", "early morning", "midday", "golden hour", "dusk", "blue hour", "night", "3am"]);
var locWeather = PN("locWeather", ["clear", "light rain", "heavy rain", "fog", "snow", "overcast", "humid heat", "windy", "thunderstorm"]);
var locMood = PN("locMood", ["cosy", "tense", "lonely", "romantic", "eerie", "bustling", "decaying", "luxurious", "sterile", "dreamy"]);
var locStyle = PN("locStyle", ["brutalist concrete", "art deco", "mid-century modern", "rustic timber", "industrial brick and steel", "minimalist white", "ornate gothic", "neon cyberpunk", "colonial stucco", "Japanese wabi-sabi", "Mughal sandstone"]);
var locProps = PN("locProps", ["a long counter running along the left wall", "a single round table at the centre", "a tall window on the right letting in the light", "a row of hanging lamps above a central aisle", "a battered sofa against the back wall", "a heavy door at the back with a small square window", "a staircase rising on the right", "stacked crates in the back-left corner", "a large mirror on the back wall", "a low wall at the front edge of frame"]);
var locMaterial = PN("locMaterial", ["scuffed oak and brass", "wet slate and black iron", "cracked plaster in faded teal", "polished concrete and chrome", "red brick and green tile", "white marble and gold trim", "weathered pine and rope", "rusted corrugated steel", "sandstone and carved jali screens", "glossy black panels and cyan light strips"]);
function locOptions(name, sub){
  let map = {locPlace, locEra, locSide, locTime, locWeather, locMood, locStyle};
  let node = map[name];
  if (!node) {
    return "";
  }
  if (name === "locPlace") {
    if (sub && node[sub]) {
      return node[sub].selectAll.map(x => String(x)).join("|||");
    }
    return node.selectAll.map(x => x.getName).join("|||");
  }
  return node.selectAll.map(x => String(x)).join("|||");

}
function locSpecText(spec){
  let s = spec || {};
  let parts = [];
  if (s.place) parts.push("Place: " + s.place + (s.category ? " (" + s.category + ")" : ""));
  if (s.side) parts.push("Interior or exterior: " + s.side);
  if (s.era) parts.push("Era: " + s.era);
  if (s.time) parts.push("Time of day: " + s.time);
  if (s.weather) parts.push("Weather: " + s.weather);
  if (s.mood) parts.push("Mood and atmosphere: " + s.mood);
  if (s.style) parts.push("Architectural or visual style: " + s.style);
  return parts.join("\n");

}
function locationRandom(spec, typeName, knownNames){
  return locationWrite("Build the location from exactly these choices. Use every one of them, do not swap any:\n" + locSpecText(spec), typeName, knownNames);

}
function locInstant(spec){
  let s = spec || {};
  let pick = n => String(n);
  let place = String(s.place || "an empty room");
  let name = place.replace(/^(a|an)\s+/i, "the ");
  let side = String(s.side || "interior");
  let p1 = pick(locProps);
  let p2 = pick(locProps);
  let tries = 0;
  while (p2 === p1 && tries < 8) {
    p2 = pick(locProps);
    tries++;
  }
  let lightMap = {dawn: "cool pale dawn light from one side", "early morning": "soft low morning sun", midday: "hard overhead daylight", "golden hour": "warm low golden sunlight raking across surfaces", dusk: "fading orange dusk light", "blue hour": "deep blue ambient light with first practical lamps on", night: "pools of warm practical light against darkness", "3am": "a few harsh practical lights, everything else in shadow"};
  let light = lightMap[s.time] || "even practical light";
  let w = String(s.weather || "clear");
  let weatherLine = side === "exterior" ? w + " weather overhead" : w + " weather visible through the windows";
  let desc = (side.charAt(0).toUpperCase() + side.slice(1)) + " " + place.replace(/^(a|an|the)\s+/i, "") + ", " + String(s.era || "present day") + ", " + String(s.style || "lived-in") + " style; " + p1 + ", " + p2 + "; " + pick(locMaterial) + " throughout; " + String(s.time || "evening") + ", " + light + "; " + weatherLine + ", a " + String(s.mood || "quiet") + " atmosphere. Camera-facing layout: facing the main wall, the first feature sits left, the second right, the entrance behind the camera.";
  return name + " || " + desc;

}
return {title, sparks, fmtStamp, clampDur, randomTypeName, randomSpark, typeMeta, t2vaRules, buildT2va, locationWrite, locationEnhance, bibleExtract, fillInT2va, locPlace, locEra, locSide, locTime, locWeather, locMood, locStyle, locProps, locMaterial, locOptions, locSpecText, locationRandom, locInstant};
})();


const PC_REF = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "MiniMax H3 Ref2VA Prompt Generator";
function fmtStamp(sec){
  let s = Number(sec);
  if (!(s > 0)) {
    s = 0;
  }
  let mm = Math.floor(s / 60);
  let ss = Math.floor(s % 60);
  let ms = Math.round((s - Math.floor(s)) * 1000);
  if (ms === 1000) {
    ss += 1;
    ms = 0;
  }
  if (ss >= 60) {
    mm += 1;
    ss = 0;
  }
  return String(mm).padStart(2, "0") + ":" + String(ss).padStart(2, "0") + "." + String(ms).padStart(3, "0");

}
function clampDur(v){
  let n = parseFloat(v);
  if (isNaN(n) || n < 1) {
    n = 1;
  }
  if (n > 15) {
    n = 15;
  }
  return n;

}
function parseCast(castJson){
  let data = {people: [], places: [], audios: []};
  try {
    let raw = JSON.parse(String(castJson || "{}"));
    if (raw && typeof raw === "object") {
      data.people = Array.isArray(raw.people) ? raw.people : [];
      data.places = Array.isArray(raw.places) ? raw.places : [];
      data.audios = Array.isArray(raw.audios) ? raw.audios : [];
    }
  } catch (e) {}
  return data;

}
function nounFor(kind){
  let k = String(kind || "person").toLowerCase();
  if (k === "woman" || k === "girl") {
    return "woman";
  }
  if (k === "man" || k === "boy") {
    return "man";
  }
  return "person";

}
function buildCastBlock(castJson){
  let data = parseCast(castJson);
  let people = data.people.slice(0, 9);
  let places = data.places.slice(0, Math.max(0, 9 - people.length));
  let audios = data.audios.slice(0, 3);
  let lines = [];
  let img = 1;
  let sub = 1;
  let map = [];
  people.forEach(function (p, i) {
    let name = String((p && p.name) || ("Person " + (i + 1))).trim();
    let noun = nounFor(p && p.kind);
    let note = String((p && p.note) || "").trim();
    lines.push("<Subject " + sub + ">: A " + noun + " with the appearance, outfit, and accessories depicted in <Image " + img + ">." + (note ? " " + note : ""));
    map.push("Image " + img + " / Subject " + sub + " = " + name + " (" + noun + ", identity plate, meta only)" + (note ? " · note: " + note : ""));
    img += 1;
    sub += 1;
  });
  places.forEach(function (p, i) {
    let name = String((p && p.name) || ("Place " + (i + 1))).trim();
    lines.push("<Subject " + sub + ">: " + name + " from <Image " + img + ">.");
    map.push("Image " + img + " / Subject " + sub + " = " + name + " (empty place plate, meta only)");
    img += 1;
    sub += 1;
  });
  audios.forEach(function (a, i) {
    let n = i + 1;
    let pIndex = parseInt(a && a.personIndex, 10);
    if (isNaN(pIndex) || pIndex < 0) {
      pIndex = 0;
    }
    if (pIndex >= people.length) {
      pIndex = 0;
    }
    let pname = people.length ? String((people[pIndex] && people[pIndex].name) || ("Person " + (pIndex + 1))) : "speaker";
    let subN = pIndex + 1;
    let timbre = String((a && a.timbre) || "voice timbre").trim();
    let label = String((a && a.label) || (pname + " voice")).trim();
    lines.push("<Audio " + n + ">: The voice-timbre reference for <Subject " + subN + "> (S" + n + ").");
    map.push("Audio " + n + " = " + label + " bound to Subject " + subN + " (" + pname + ") as (S" + n + "). Timbre meta: " + timbre + ". No real audio file in this helper.");
  });
  return {
    defs: lines.join("\n"),
    map: map.join("\n"),
    imageCount: img - 1,
    audioCount: audios.length,
    personCount: people.length,
    placeCount: places.length
  };

}
function buildRef2va(castJson, interaction, dur, language){
  dur = clampDur(dur);
  language = String(language || "English");
  let silent = language === "none";
  let endTs = fmtStamp(dur);
  let durLabel = String(Math.round(dur * 10) / 10) + "s";
  let t = String(interaction || "").trim();
  if (!t) {
    t = "(empty — write a short interaction using only the listed people and places; no extra cast)";
  }
  let cast = buildCastBlock(castJson);
  let talk = silent ? "No spoken dialogue. No (Sx). No <d> blocks. No <Audio N> performance. Intertitles in English double quotes if needed." : ("DIALOGUE LANGUAGE: " + language + ". Every spoken line MUST be written in " + language + " inside <d>[" + language + "] ...</d>. Body English. If the user wrote lines in another language, translate them into " + language + " inside <d>. Speaking referenced people: <Subject N> (Sx) says, <d>[" + language + "] ...</d>. Cite <Audio N> on the first vocal beat for that speaker: speaking in the voice timbre from <Audio N>. Voiceover: says in an off-screen voiceover: <d>[" + language + "] ...</d> while lips remain completely closed. One speaking face per shot, front-facing, medium or closer. Silent people get no (Sx).");
  let audioSum = cast.audioCount > 0 ? " + audio reference" : "";
  return "Rewrite the user interaction into a MiniMax-H3 Ref2VA prompt (VIDEO_PROMPT_WRITING_GUIDE_ref_en.md plus base shot grammar). Output exactly these six sections and nothing else (no intro, no markdown fences, no attach preface):\n\nsubject_definitions:\n" + (cast.defs || "<Subject 1>: A person with the appearance, outfit, and accessories depicted in <Image 1>.") + "\n\nsummary:\n[reference generation" + audioSum + "] ...\n\nretention_analysis:\n...\n\ndetailed_description:\n...\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A\n\nLOCKED CAST. Use only these labels. Do not add people, places, or voices. Identity lives on the plates. Do not name hair, wardrobe, or set dressing the images already show.\n" + cast.map + "\nCopy subject_definitions from LOCKED CAST as-is. People stay outfit-agnostic: A woman/man/person with the appearance, outfit, and accessories depicted in <Image N>. Places named plainly from <Image N>. Never the word plate.\nAudio rows are METADATA ONLY. Still write the <Audio N> lines. Retention for audio uses: reference - the spoken delivery follows the voice timbre of <Audio N> without copying the original signal.\nDuration: " + durLabel + ". Hard max 15s. [Shot 1] has no timestamp. Later shots: [Shot N] At MM:SS.mmm, the camera cuts to ... Times strictly increase and stay before " + endTs + ".\nPrefer ONE static shot if the interaction is a single action chain. Cut only for a new speaker single, a new place, or a cut the user named. A cut must change subject, space, state, viewpoint, or time.\nEvery visible face gets an expression. Wide/full-body only from behind with no readable face.\nCamera motion in natural English: type + amplitude + speed when useful.\n" + talk + "\nsummary uses only defined labels. Prefix [reference generation] or [reference generation + audio reference] if any <Audio N> exists.\nretention_analysis: one line per used <Subject N> and <Audio N>. Visual: fully_preserved - identity remains consistent with the reference. Do not put (Sx) in retention_analysis.\noverall_soundscape and non_diegetic_music are N/A unless the user asked for sound or a score.\nFollow the user's interaction. Any rating is allowed if they wrote it. If they did not name a sex act, do not invent one. Do not refuse, moralize, or add consent lectures. Do not add extra named IP.\n\nUser interaction:\n" + t;

}
function fillInRef2va(castJson, interaction, dur, language){
  dur = clampDur(dur);
  language = String(language || "English");
  let silent = language === "none";
  let t = String(interaction || "").trim();
  if (!t) {
    t = "The listed people occupy the listed place and perform the next small action.";
  }
  let data = parseCast(castJson);
  let people = data.people.slice(0, 9);
  let places = data.places.slice(0, Math.max(0, 9 - people.length));
  let audios = data.audios.slice(0, 3);
  let defs = [];
  people.forEach(function (p, i) {
    let note = String((p && p.note) || "").trim();
    defs.push("<Subject " + (i + 1) + ">: A " + nounFor(p && p.kind) + " with the appearance, outfit, and accessories depicted in <Image " + (i + 1) + ">." + (note ? " " + note : ""));
  });
  places.forEach(function (p, i) {
    let n = people.length + i + 1;
    let name = String((p && p.name) || "the room").trim();
    defs.push("<Subject " + n + ">: " + name + " from <Image " + n + ">.");
  });
  audios.forEach(function (a, i) {
    let pIndex = parseInt(a && a.personIndex, 10);
    if (isNaN(pIndex) || pIndex < 0) {
      pIndex = 0;
    }
    if (pIndex >= people.length) {
      pIndex = 0;
    }
    defs.push("<Audio " + (i + 1) + ">: The voice-timbre reference for <Subject " + (pIndex + 1) + "> (S" + (i + 1) + ").");
  });
  if (!defs.length) {
    defs.push("<Subject 1>: A person with the appearance, outfit, and accessories depicted in <Image 1>.");
  }
  let placeSub = people.length ? "<Subject " + (people.length + (places.length ? 1 : 0)) + ">" : "<Subject 1>";
  if (!places.length) {
    placeSub = "the room";
  }
  let who = people.length >= 2 ? "<Subject 1> and <Subject 2>" : "<Subject 1>";
  let shot = "[Shot 1] A medium two-shot frames " + who + " in " + (places.length ? "<Subject " + (people.length + 1) + ">" : "the room") + ". " + t + " The camera holds a static shot. Faces look intent.";
  if (!silent && audios.length && language !== "none") {
    shot += " <Subject 1> (S1) says, <d>[" + language + "] ...</d>";
  }
  let ret = [];
  let lastSub = people.length + places.length;
  if (lastSub < 1) {
    lastSub = 1;
  }
  for (let i = 1; i <= lastSub; i++) {
    ret.push("<Subject " + i + "> (appears in [Shot 1]): fully_preserved - identity remains consistent with the reference.");
  }
  audios.forEach(function (a, i) {
    ret.push("<Audio " + (i + 1) + ">: reference - the spoken delivery follows the voice timbre of <Audio " + (i + 1) + "> without copying the original signal.");
  });
  let prefix = audios.length ? "[reference generation + audio reference]" : "[reference generation]";
  return "subject_definitions:\n" + defs.join("\n") + "\n\nsummary:\n" + prefix + " " + who + " interact as described. No extra cast.\n\nretention_analysis:\n" + ret.join("\n") + "\n\ndetailed_description:\nLive-action, cinematic, fine film grain.\n" + shot + "\n\noverall_soundscape:\nN/A\n\nnon_diegetic_music:\nN/A";

}
return {title, fmtStamp, clampDur, parseCast, nounFor, buildCastBlock, buildRef2va, fillInRef2va};
})();


const PC_SOUND = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "MiniMax H3 Soundscape & Music Writer";
var soundOrder = PN("soundOrder", ["auto", "none", "bollywood1980s", "noir", "horror", "interior", "monsoon", "temple", "kitchen", "disco", "beach", "street", "train", "office", "wedding"]);
function soundMeta(name){
  let d = {
    auto: {label: "auto (from scene)", bits: "room tone matched to the scene"},
    none: {label: "silence — N/A", bits: ""},
    bollywood1980s: {label: "1980s Bombay", hint: "1980s Bombay ambience — distant traffic, a ceiling fan, crows, a neighbour's radio", bits: "distant traffic, a ceiling fan, crows, and a neighbour's radio"},
    noir: {label: "film noir", hint: "film noir ambience — rain on glass, a ticking clock, a distant siren, heels on wet pavement", bits: "rain on glass, a ticking clock, a distant siren, and heels on wet pavement"},
    horror: {label: "horror", hint: "horror ambience — a low room tone, a settling house, a dripping pipe, wind through a crack", bits: "a low room tone, a settling house, a dripping pipe, and wind through a crack"},
    interior: {label: "domestic interior", hint: "domestic interior ambience — a ceiling fan, kitchen clatter one room away, cloth, a ticking clock", bits: "a ceiling fan, kitchen clatter one room away, cloth, and a ticking clock"},
    monsoon: {label: "monsoon rain", hint: "monsoon ambience — heavy rain on a tin roof, thunder far off, water in a gutter", bits: "heavy rain on a tin roof, thunder far off, and water in a gutter"},
    temple: {label: "temple / ritual", hint: "temple ambience — a bell, murmured prayer, incense, distant chant", bits: "a bell, murmured prayer, and distant chant"},
    kitchen: {label: "kitchen / cooking", hint: "kitchen ambience — oil in a pan, a pressure-cooker hiss, a knife on a board, a running tap", bits: "oil in a pan, a pressure-cooker hiss, a knife on a board, and a running tap"},
    disco: {label: "nightclub / disco", hint: "nightclub ambience — a crowd murmur, bass through a wall, glasses, a four-on-the-floor beat", bits: "a crowd murmur, bass through a wall, glasses, and a four-on-the-floor beat"},
    beach: {label: "beach / shore", hint: "beach ambience — surf, wind in palms, gulls, a boat motor far off", bits: "surf, wind in palms, gulls, and a boat motor far off"},
    street: {label: "city street", hint: "street ambience — horns, a bus door, vendors, footsteps on pavement", bits: "horns, a bus door, vendors, and footsteps on pavement"},
    train: {label: "train compartment", hint: "rail-compartment ambience — wheel rhythm, a berth fan, a vendor in the corridor", bits: "wheel rhythm, a berth fan, and a vendor in the corridor"},
    office: {label: "office", hint: "office ambience — a ceiling fan, paper, a distant typewriter, corridor footsteps", bits: "a ceiling fan, paper, a distant typewriter, and corridor footsteps"},
    wedding: {label: "wedding", hint: "wedding ambience — a dhol outside, chatter, glass, firecrackers far off", bits: "a dhol outside, chatter, glass, and firecrackers far off"}
  };
  return d[name] || d.auto;

}
var musicTypeOrder = PN("musicTypeOrder", ["none", "bgm", "song", "remix", "source"]);
function musicTypeMeta(name){
  let d = {
    none: {label: "N/A (default)"},
    bgm: {label: "BGM — background score"},
    song: {label: "New song — with lyrics"},
    remix: {label: "Remix — rework of a tune"},
    source: {label: "Source music — radio/TV/band"}
  };
  return d[name] || d.none;

}
var genreOrder = PN("genreOrder", ["auto", "filmi", "classical", "suspense", "strings", "tabla", "jazz", "disco", "rock", "lofi", "qawwali", "ghazal", "synthwave", "folk", "hiphop", "electronic"]);
function genreMeta(name){
  let d = {
    auto: {label: "auto (from scene)", hint: "instrumentation the scene implies"},
    filmi: {label: "Bollywood filmi", hint: "Bollywood filmi — a string section, dholak, a sung melody"},
    classical: {label: "classical orchestral", hint: "classical orchestral — strings and woodwinds"},
    suspense: {label: "suspense synth", hint: "suspense synth — a low pulse and held tones"},
    strings: {label: "romantic strings", hint: "strings — slow bowed phrases"},
    tabla: {label: "tabla / percussion", hint: "tabla and light percussion"},
    jazz: {label: "jazz combo", hint: "jazz combo — brushed drums, upright bass, a muted horn"},
    disco: {label: "disco", hint: "disco — a four-on-the-floor kick, hi-hats, a bassline"},
    rock: {label: "rock", hint: "rock — electric guitar, bass, a steady backbeat"},
    lofi: {label: "lofi piano", hint: "lofi piano — soft keys and a slow beat"},
    qawwali: {label: "qawwali", hint: "qawwali — harmonium, tabla, handclaps, call and response"},
    ghazal: {label: "ghazal", hint: "ghazal — santoor, sarangi, a slow measured vocal"},
    synthwave: {label: "synthwave", hint: "synthwave — analog pads, a pulsing bass, gated drums"},
    folk: {label: "folk", hint: "folk — acoustic guitar, a flute, hand percussion"},
    hiphop: {label: "hip-hop", hint: "hip-hop — a looped beat, a deep bassline, a scratched sample"},
    electronic: {label: "electronic", hint: "electronic — layered synths and a steady or broken beat"}
  };
  return d[name] || d.auto;

}
var tempoOrder = PN("tempoOrder", ["auto", "slow", "medium", "upbeat", "fast", "frantic"]);
function tempoMeta(name){
  let d = {
    auto: {label: "auto (from scene)", hint: "tempo the scene implies"},
    slow: {label: "slow", hint: "slow tempo"},
    medium: {label: "medium", hint: "medium tempo"},
    upbeat: {label: "upbeat", hint: "upbeat tempo"},
    fast: {label: "fast", hint: "fast tempo"},
    frantic: {label: "frantic", hint: "frantic tempo"}
  };
  return d[name] || d.auto;

}
var eraOrder = PN("eraOrder", ["auto", "seventies", "eighties", "nineties", "modern", "timeless"]);
function eraMeta(name){
  let d = {
    auto: {label: "auto (from scene)", hint: "the era the scene implies"},
    seventies: {label: "1970s", hint: "1970s production — live strings, a roomy vocal, analog tape"},
    eighties: {label: "1980s", hint: "1980s production — analog synths, gated drums, cassette warmth"},
    nineties: {label: "1990s", hint: "1990s production — drum machines, bright keys, a clean mix"},
    modern: {label: "modern", hint: "modern production — a crisp, wide mix"},
    timeless: {label: "timeless", hint: "a timeless acoustic ensemble"}
  };
  return d[name] || d.auto;

}
var vocalOrder = PN("vocalOrder", ["female", "male", "duet", "chorus", "hum", "instrumental"]);
function vocalMeta(name){
  let d = {
    female: {label: "female vocal", hint: "a female lead vocal", lyrics: true},
    male: {label: "male vocal", hint: "a male lead vocal", lyrics: true},
    duet: {label: "duet", hint: "a male and female duet", lyrics: true},
    chorus: {label: "chorus", hint: "a full chorus", lyrics: true},
    hum: {label: "hummed", hint: "a hummed melody with no lyric", lyrics: false},
    instrumental: {label: "instrumental", hint: "no vocals", lyrics: false}
  };
  return d[name] || d.instrumental;

}
var langOrder = PN("langOrder", ["English", "Hindi", "Hinglish", "Kannada", "Tamil", "Urdu"]);
var durOrder = PN("durOrder", ["s6", "s10", "s15"]);
function durMeta(name){
  let d = {
    s6: {label: "6 seconds", seconds: 6},
    s10: {label: "10 seconds", seconds: 10},
    s15: {label: "15 seconds", seconds: 15}
  };
  if (typeof name === "number" && isFinite(name)) {
    let s = Math.round(name);
    if (s < 1) {
      s = 1;
    }
    if (s > 600) {
      s = 600;
    }
    return {label: durLabel(s), seconds: s};
  }
  if (typeof name === "string") {
    let t = name.trim().toLowerCase();
    let m = t.match(/^s(\d+(?:\.\d+)?)$/);
    if (m) {
      return durMeta(Number(m[1]));
    }
    m = t.match(/^(\d+(?:\.\d+)?)\s*s(?:ec(?:ond)?s?)?$/);
    if (m) {
      return durMeta(Number(m[1]));
    }
    m = t.match(/^(\d+(?:\.\d+)?)\s*m(?:in(?:ute)?s?)?$/);
    if (m) {
      return durMeta(Number(m[1]) * 60);
    }
    m = t.match(/^(\d+)\s*:\s*(\d{1,2})$/);
    if (m) {
      return durMeta(Number(m[1]) * 60 + Number(m[2]));
    }
    let n = Number(t);
    if (isFinite(n) && t !== "") {
      return durMeta(n);
    }
  }
  return d[name] || d.s10;

}
function clipSeconds(v){
  let d = null;
  try {
    d = durMeta(v);
  } catch (e) {
    d = null;
  }
  return (d && d.seconds) ? d.seconds : 10;

}
function durLabel(seconds){
  let s = Math.round(Number(seconds));
  if (!isFinite(s) || s < 1) {
    s = 10;
  }
  if (s > 600) {
    s = 600;
  }
  if (s < 60) {
    return s + " seconds";
  }
  let m = Math.floor(s / 60);
  let r = s % 60;
  if (r === 0) {
    return s + " seconds (" + m + "m)";
  }
  return s + " seconds (" + m + "m " + r + "s)";

}
function durShort(seconds){
  let s = Math.round(Number(seconds));
  if (!isFinite(s) || s < 1) {
    s = 10;
  }
  if (s > 600) {
    s = 600;
  }
  if (s < 60) {
    return s + "s";
  }
  let m = Math.floor(s / 60);
  let r = s % 60;
  if (r === 0) {
    return m + "m";
  }
  return m + "m " + r + "s";

}
function durFileBit(seconds){
  let s = Math.round(Number(seconds));
  if (!isFinite(s) || s < 1) {
    s = 10;
  }
  if (s > 600) {
    s = 600;
  }
  if (s < 60) {
    return s + "s";
  }
  let m = Math.floor(s / 60);
  let r = s % 60;
  if (r === 0) {
    return m + "m";
  }
  return m + "m" + r + "s";

}
function soundSpan(seconds){
  seconds = Number(seconds);
  if (seconds <= 6) {
    return "1-2";
  }
  if (seconds <= 10) {
    return "2-3";
  }
  if (seconds <= 30) {
    return "2-4";
  }
  if (seconds <= 60) {
    return "3-5";
  }
  if (seconds <= 180) {
    return "4-7";
  }
  return "5-9";

}
function lyricSpan(seconds){
  seconds = Number(seconds);
  if (seconds <= 6) {
    return "2-4";
  }
  if (seconds <= 10) {
    return "4-6";
  }
  if (seconds <= 30) {
    return "6-10";
  }
  if (seconds <= 60) {
    return "8-14";
  }
  if (seconds <= 180) {
    return "12-20";
  }
  return "16-30";

}
function sungLyrics(vocalName){
  let vm = vocalMeta(vocalName);
  return !(vm && vm.lyrics === false);

}
function optionsFor(order, metaFn){
  let out = [];
  let keys = [];
  if (!order) {
    keys = [];
  } else if (order.selectAll) {
    let arr = order.selectAll;
    if (typeof arr === "function") {
      arr = arr();
    }
    for (let i = 0; i < arr.length; i++) {
      let it = arr[i];
      let k = it;
      if (it != null && it.evaluateItem !== undefined) {
        try {
          k = it.evaluateItem;
        } catch (e) {
          k = String(it);
        }
      }
      k = String(k).trim();
      if (k) {
        keys.push(k);
      }
    }
  } else if (Array.isArray(order)) {
    keys = order.map(function (x) { return String(x).trim(); }).filter(Boolean);
  } else if (typeof order === "string") {
    keys = order.split(/\s+/).map(function (x) { return x.trim(); }).filter(Boolean);
  } else {
    keys = [String(order).trim()].filter(Boolean);
  }
  keys.forEach(function (k) {
    let m = null;
    try {
      m = metaFn ? metaFn(k) : null;
    } catch (e) {
      m = null;
    }
    out.push(k + "=" + ((m && m.label) || k));
  });
  return out.join("|||");

}
function soundOptions(){
  return optionsFor(soundOrder, soundMeta);

}
function musicTypeOptions(){
  return optionsFor(musicTypeOrder, musicTypeMeta);

}
function genreOptions(){
  return optionsFor(genreOrder, genreMeta);

}
function tempoOptions(){
  return optionsFor(tempoOrder, tempoMeta);

}
function eraOptions(){
  return optionsFor(eraOrder, eraMeta);

}
function vocalOptions(){
  return optionsFor(vocalOrder, vocalMeta);

}
function langOptions(){
  return optionsFor(langOrder, null);

}
function durOptions(){
  return optionsFor(durOrder, durMeta);

}
function atTempo(tempoName){
  let tm = tempoMeta(tempoName);
  let hint = (tm && tm.hint) ? tm.hint : "medium tempo";
  if (!tempoName || tempoName === "auto") {
    hint = "medium tempo";
  }
  hint = String(hint).replace(" tempo", "");
  let article = /^[aeiou]/i.test(hint) ? "an" : "a";
  return "at " + article + " " + hint + " tempo";

}
function lyricStub(lang, seconds){
  let bank = {
    English: ["The fan keeps time with the rain", "The door stays a little open", "The street is louder than the room", "The wait stays in this room", "Wet steps still catch the light", "A name stops in the air"],
    Hindi: ["रात का पंखा बारिश के साथ चलता है", "दरवाज़ा थोड़ा खुला रह गया", "सड़क कमरे से ज़्यादा शोर करती है", "इंतज़ार यहीं रुक जाता है", "गीली सीढ़ियाँ अभी चमक रही हैं", "हवा में एक नाम रुक जाता है"],
    Hinglish: ["Raat ka pankha baarish ke saath chalta hai", "Darwaza thoda khula reh gaya", "Sadak kamre se zyada shor karti hai", "Intezar yahin ruk jaata hai", "Geeli seedhiyan ab bhi chamak rahi hain", "Hawa mein ek naam ruk jaata hai"],
    Kannada: ["ಮಳೆ ಬೀಳುತ್ತಿದೆ", "ಫ್ಯಾನ್ ತಿರುಗುತ್ತಿದೆ", "ಬಾಗಿಲು ತೆರೆದಿದೆ", "ನಾನು ಇಲ್ಲೇ ಇದ್ದೇನೆ", "ಮಳೆ ನಿಲ್ಲಲಿಲ್ಲ", "ದೀಪ ಉರಿಯುತ್ತಿದೆ"],
    Tamil: ["மழை பெய்கிறது", "விசிறி சுற்றுகிறது", "கதவு திறந்துள்ளது", "நான் இங்கே இருக்கிறேன்", "மழை நிற்கவில்லை", "விளக்கு எரிகிறது"],
    Urdu: ["رات کا پنکھا بارش کے ساتھ چلتا ہے", "دروازہ تھوڑا کھلا رہ گیا", "گلی کمرے سے زیادہ شور کرتی ہے", "انتظار یہیں رک جاتا ہے", "گیلی سیڑھیاں اب بھی چمک رہی ہیں", "ہوا میں ایک نام رک جاتا ہے"]
  };
  let lines = bank[lang] || bank.English;
  seconds = Number(seconds);
  let n = 4;
  if (seconds <= 6) {
    n = 2;
  } else if (seconds <= 10) {
    n = 4;
  } else if (seconds <= 30) {
    n = 8;
  } else if (seconds <= 60) {
    n = 12;
  } else if (seconds <= 180) {
    n = 16;
  } else {
    n = 24;
  }
  let picked = [];
  for (let i = 0; i < n; i++) {
    picked.push(lines[i % lines.length]);
  }
  return "<d>[" + (bank[lang] ? lang : "English") + "]\n" + picked.join("\n") + "\n</d>";

}
function sourceNoun(genreName){
  if (genreName === "disco" || genreName === "rock" || genreName === "electronic") {
    return "speaker";
  }
  if (genreName === "hiphop") {
    return "passing speaker";
  }
  if (genreName === "filmi") {
    return "television";
  }
  if (genreName === "qawwali" || genreName === "folk") {
    return "courtyard";
  }
  return "radio";

}
function buildSound(scene, soundPreset, musicType, genreName, tempoName, eraName, vocalName, langName, durName){
  let sm = soundMeta(soundPreset);
  let gm = genreMeta(genreName);
  let em = eraMeta(eraName);
  let vm = vocalMeta(vocalName);
  let seconds = clipSeconds(durName);
  let lang = String(langName || "English");
  let s = String(scene || "").trim();
  if (!s) {
    s = "a short scene in a single location";
  }
  let wantsSong = musicType === "song" || musicType === "remix";
  let wantsSource = musicType === "source";
  let sung = sungLyrics(vocalName);
  let genreHint = (gm && gm.hint) ? gm.hint : "instrumentation the scene implies";
  let eraHint = (em && em.hint && eraName !== "auto") ? em.hint : "a timeless ensemble";
  let vocalHint = (vm && vm.hint) ? vm.hint : "no vocals";
  let tempoPhrase = atTempo(tempoName);
  let fields = "overall_soundscape: ...\n\nnon_diegetic_music: ...";
  if (wantsSong) {
    fields += "\n\nsong: ...";
  }
  if (wantsSource) {
    fields += "\n\ndiegetic_source_music: ...";
  }
  let rules = "Output exactly these fields and nothing else. No intro, no markdown fences, no labels about where to paste, and no field names inside the values:\n\n" + fields + "\n\n";
  rules += "This is MiniMax-H3 base format (VIDEO_PROMPT_WRITING_GUIDE_base_en.md). Clip length: " + durLabel(seconds) + ". Keep the sound inside that length.\n";
  rules += "overall_soundscape: " + soundSpan(seconds) + " English sentences of ambience and physical, non-verbal sounds only — room tone and foley such as footsteps, cloth, doors, water, and traffic. Never quote dialogue, lyrics, or on-screen text. Never put a <d> tag in overall_soundscape.\n";
  if (soundPreset === "none" && !wantsSource) {
    rules += "The user chose silence for ambience: overall_soundscape must be exactly N/A.\n";
  } else if (soundPreset === "none" && wantsSource) {
    rules += "The user chose no extra ambience. overall_soundscape is only the source music the characters hear, in one or two sentences. Do not write N/A.\n";
  } else {
    rules += "Soundscape register: " + ((sm && sm.hint) ? sm.hint : "the space the scene implies") + ".\n";
  }
  if (musicType === "none") {
    rules += "non_diegetic_music must be exactly N/A. Do not invent a score, a song, or source music.\n";
  } else if (musicType === "bgm") {
    rules += "non_diegetic_music: 1-3 sentences. Name the instruments, the tempo, the rhythm, and the dynamics (enter, hold, swell, fade). No abstract mood words (romantic, tense, eerie, sad, epic, nostalgic, dramatic, emotional, haunting, ominous, melancholic, hopeful). Genre: " + genreHint + ". Tempo: " + tempoPhrase + ". Era: " + eraHint + ".\n";
  } else if (musicType === "song") {
    rules += "A NEW ORIGINAL SONG. Do not use a known title or a copyrighted lyric. non_diegetic_music must be exactly N/A. The song value is one or two English sentences of performance (who sings, vocal style, instruments, speed, rhythm, dynamics)";
    if (sung) {
      rules += ", then original lyrics inside <d>[" + lang + "] ... </d>. Lyrics: " + lyricSpan(seconds) + " short lines in " + lang + ", matched to the scene.\n";
    } else {
      rules += ". No <d> block and no lyric lines.\n";
    }
    rules += "Vocal: " + vocalHint + ". Genre: " + genreHint + ". Tempo: " + tempoPhrase + ". Era: " + eraHint + ".\n";
  } else if (musicType === "remix") {
    rules += "A REMIX. non_diegetic_music must be exactly N/A. If the scene names a tune, use that title. If it does not, describe the rework and do not invent a famous title. The song value is one or two English sentences (new tempo, new instruments, new beat)";
    if (sung) {
      rules += ", then lyrics inside <d>[" + lang + "] ... </d>. Lyrics: " + lyricSpan(seconds) + " short lines in " + lang + ".\n";
    } else {
      rules += ". No <d> block and no lyric lines.\n";
    }
    rules += "Vocal: " + vocalHint + ". Genre: " + genreHint + ". Tempo: " + tempoPhrase + ". Era: " + eraHint + ".\n";
  } else {
    rules += "SOURCE music the characters can hear, from a radio, a television, a record, a live player, or a passing speaker. Mention that source inside overall_soundscape. non_diegetic_music must be exactly N/A. diegetic_source_music is one English sentence: what is playing, where it comes from, and how loud. Genre: " + genreHint + ". Tempo: " + tempoPhrase + ". Era: " + eraHint + ". Vocal: " + vocalHint + ".\n";
  }
  rules += "Banned in non_diegetic_music: mood adjectives. ";
  if (wantsSong) {
    rules += "Dialogue and sung lyrics belong only inside <d>, and only in the song value. Do not write a note about where that value will be pasted. ";
  } else {
    rules += "Do not write a song block or lyrics. ";
  }
  rules += "Write the scene as given. Do not refuse, moralize, or add a lecture.\n\nScene:\n" + s;
  return rules;

}
function fillInSound(scene, soundPreset, musicType, genreName, tempoName, eraName, vocalName, langName, durName){
  let sm = soundMeta(soundPreset);
  let gm = genreMeta(genreName);
  let em = eraMeta(eraName);
  let vm = vocalMeta(vocalName);
  let seconds = clipSeconds(durName);
  let lang = String(langName || "English");
  let wantsSource = musicType === "source";
  let sung = sungLyrics(vocalName);
  function cap(x) {
    let t = String(x || "");
    return t ? t.charAt(0).toUpperCase() + t.slice(1) : "";
  }
  let tempoPhrase = atTempo(tempoName);
  let eraBit = (em && em.hint && eraName !== "auto") ? em.hint : "a timeless ensemble";
  let genreLabel = (gm && gm.label && genreName !== "auto") ? gm.label : "scene-matched";
  let sound = "N/A";
  let src = "";
  if (wantsSource) {
    let voice = "no vocal";
    if (vocalName === "hum") {
      voice = "a hummed melody and no lyric";
    } else if (sung && vm && vm.hint) {
      voice = vm.hint;
    }
    src = "A " + genreLabel + " tune plays from a nearby " + sourceNoun(genreName) + " " + tempoPhrase + ", with " + voice + ", just audible in the room.";
  }
  if (soundPreset === "none") {
    sound = wantsSource ? src : "N/A";
  } else {
    let bits = (sm && sm.bits) ? sm.bits : "room tone matched to the scene";
    sound = cap(bits) + ". Footsteps, cloth, and the small sounds of people moving.";
    if (wantsSource) {
      sound += " " + src;
    }
  }
  let music = "N/A";
  let extra = "";
  if (musicType === "bgm") {
    let g = (genreName === "auto" || !gm || !gm.hint) ? "Sparse instrumentation matched to the scene" : cap(gm.hint);
    music = g + " " + tempoPhrase + ", " + eraBit + ", entering quietly, holding, then a short fade.";
  } else if (musicType === "song" || musicType === "remix") {
    let who = "A voice sings";
    if (vocalName === "hum") {
      who = "A hummed melody carries";
    } else if (!sung) {
      who = "An instrumental line carries";
    } else if (vm && vm.hint) {
      who = cap(vm.hint) + " sings";
    }
    let kind = musicType === "remix" ? "reworked" : "original";
    let article = kind === "original" ? "an" : "a";
    extra = "song: " + who + " " + article + " " + kind + " " + genreLabel + " tune " + tempoPhrase + ", " + eraBit + ".";
    if (sung) {
      extra += "\n" + lyricStub(lang, seconds);
    }
  }
  if (wantsSource && src) {
    extra = "diegetic_source_music: " + src;
  }
  let out = "overall_soundscape: " + sound + "\n\nnon_diegetic_music: " + music;
  if (extra) {
    out += "\n\n" + extra;
  }
  return out;

}
return {title, soundOrder, soundMeta, musicTypeOrder, musicTypeMeta, genreOrder, genreMeta, tempoOrder, tempoMeta, eraOrder, eraMeta, vocalOrder, vocalMeta, langOrder, durOrder, durMeta, clipSeconds, durLabel, durShort, durFileBit, soundSpan, lyricSpan, sungLyrics, optionsFor, soundOptions, musicTypeOptions, genreOptions, tempoOptions, eraOptions, vocalOptions, langOptions, durOptions, atTempo, lyricStub, sourceNoun, buildSound, fillInSound};
})();


const PC_ORBIT = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "Center Orbit \u2014 360 T2VA";
function clipSeconds(v){
  let n = parseFloat(v);
  if (n === 6 || n === 10 || n === 15) return n;
  return 10;

}
function clean(s){
  return String(s || "").replace(/\s+/g, " ").trim();

}
function cap(x){
  let t = clean(x);
  return t ? t.charAt(0).toUpperCase() + t.slice(1) : "";

}
function nameCore(name){
  return clean(name).replace(/^(a|an|the|one|two|three|four)\s+/i, "").toLowerCase();

}
function asSubject(name){
  let t = clean(name);
  if (/^(one|two|three|four)\b/i.test(t)) return cap(t);
  t = t.replace(/^(a|an|the)\s+/i, "");
  return "The " + t;

}
function sitLine(name, where){
  let subject = asSubject(name);
  if (subject.indexOf(",") !== -1) return subject + ", sits " + where;
  return subject + " sits " + where;

}
function fillsLine(name){
  let subject = asSubject(name);
  if (subject.indexOf(",") !== -1) return subject + ", fills the centre of the frame";
  return subject + " fills the centre of the frame";

}
function cueWord(word){
  return /^(red|blue|green|yellow|black|white|brown|grey|gray|gold|golden|pink|orange|purple|maroon|cream|ivory|teal|saffron|ochre|ocher|faded|brass|tin|glass|rope|cloth|cotton|silk|iron|steel|copper|cane|bamboo|leather|ceramic|clay|stone|marble|teak|velvet|striped|checked|checkered|carved|painted|rusted|wooden|wood|plastic|chrome|enamel|wicker|mud)$/i.test(word);

}
function headNoun(name){
  let t = clean(name).toLowerCase().split(",")[0];
  t = t.split(/\b(with|hanging|against|under|over|beside|from)\b/)[0];
  let words = t.replace(/[^a-z\s-]/g, " ").trim().split(/\s+/);
  let fillers = {a: 1, an: 1, the: 1, one: 1, two: 1, three: 1, four: 1, small: 1, large: 1, big: 1, little: 1, square: 1, round: 1, long: 1, low: 1, high: 1, old: 1, new: 1, open: 1, shut: 1, bare: 1, pale: 1};
  let i = words.length - 1;
  while (i > 0 && (fillers[words[i]] || cueWord(words[i]))) i = i - 1;
  return words[i] || "";

}
function leftOf(dir){
  let walls = ["north", "east", "south", "west"];
  let idx = {north: 0, east: 1, south: 2, west: 3};
  return walls[(idx[dir] + 3) % 4];

}
function rightOf(dir){
  let walls = ["north", "east", "south", "west"];
  let idx = {north: 0, east: 1, south: 2, west: 3};
  return walls[(idx[dir] + 1) % 4];

}
function cornerKey(a, b){
  let set = {};
  set[a] = true;
  set[b] = true;
  if (set.north && set.east) return "ne";
  if (set.north && set.west) return "nw";
  if (set.south && set.east) return "se";
  if (set.south && set.west) return "sw";
  return "";

}
function wallPhrase(name, dir){
  let t = clean(name);
  if (t) return t;
  return "the bare " + dir + " wall";

}
function slotMap(north, east, south, west, ne, nw, se, sw, middle){
  return {
    north: wallPhrase(north, "north"),
    east: wallPhrase(east, "east"),
    south: wallPhrase(south, "south"),
    west: wallPhrase(west, "west"),
    ne: clean(ne),
    nw: clean(nw),
    se: clean(se),
    sw: clean(sw),
    middle: clean(middle)
  };

}
function viewLine(face, slots){
  let left = leftOf(face);
  let right = rightOf(face);
  let lc = cornerKey(face, left);
  let rc = cornerKey(face, right);
  let parts = [];
  parts.push(fillsLine(slots[face]) + ".");
  parts.push(sitLine(slots[left], "on the left") + ".");
  parts.push(sitLine(slots[right], "on the right") + ".");
  if (slots[lc]) parts.push(sitLine(slots[lc], "in the left background") + ".");
  if (slots[rc]) parts.push(sitLine(slots[rc], "in the right background") + ".");
  return parts.join(" ");

}
function middleLevel(middle, where){
  let name = clean(middle);
  if (!name) return "";
  let w = String(where || "above");
  if (w === "floor") return sitLine(name, "on the floor around the camera, along the bottom edge of the frame") + ".";
  if (w === "waist") return sitLine(name, "at waist height around the camera, across the bottom of the frame") + ".";
  return sitLine(name, "directly above the camera, out of the top of the frame") + ".";

}
function middleVertical(middle, where, dir){
  let name = clean(middle);
  if (dir === "up") {
    if (name && String(where || "above") === "above") return fillsLine(name) + ".";
    if (name) return asSubject(name) + " stays below the frame.";
    return "The bare ceiling fills the frame.";
  }
  if (name && where === "floor") return fillsLine(name) + ".";
  if (name && where === "waist") return sitLine(name, "around the camera at the bottom of the frame") + ".";
  if (name) return asSubject(name) + " stays above the frame.";
  return "The bare floor fills the frame.";

}
function passCorner(key, slots){
  if (!slots[key]) return "";
  let label = {ne: "northeast", nw: "northwest", se: "southeast", sw: "southwest"};
  return "The " + label[key] + " corner comes through next. " + sitLine(slots[key], "in that corner") + ".";

}
function styleOpen(style){
  let t = clean(style);
  if (!t) t = "Live-action, cinematic";
  return t.replace(/\.+$/, "");

}
function placeSentence(place){
  let t = clean(place);
  if (!t) t = "A small empty room, about 3 by 3 metres";
  if (!/[.!?]$/.test(t)) t += ".";
  return t;

}
function lockLine(){
  return "No people, no faces, no mannequins, no text, and no watermark. Each fixed object keeps its own color, material, and shape. The same room, the same furniture, and the same light stay fixed.";

}
function soundBed(sound, light){
  let s = clean(sound);
  if (s) {
    if (!/[.!?]$/.test(s)) s += ".";
    return s;
  }
  let l = clean(light);
  if (l) return l.replace(/\.+$/, "") + ", heard only as a faint practical. No voices.";
  return "Room tone. No voices.";

}
function lead(style, place, camera){
  return "[Shot 1] " + styleOpen(style) + ", " + camera + " " + placeSentence(place);

}
function describeSweep(dur, style, place, slots, where){
  let secs = clipSeconds(dur);
  let q = Math.round((secs / 4) * 10) / 10;
  let parts = [];
  parts.push(lead(style, place, "a medium-wide shot from a camera standing in the exact centre of the room, at eye height, facing the north wall."));
  parts.push("This is a fully enclosed room with four solid walls: north ahead, east to the right, south directly behind the camera, and west to the left. Every wall is a real wall of the room, not an open set.");
  parts.push(fillsLine(slots.north) + ".");
  let mid = middleLevel(slots.middle, where);
  if (mid) parts.push(mid);
  parts.push("The camera stays on the same spot and rotates in place to the right, a full 360 degrees, in one continuous seamless turn at constant slow speed, with no cuts and no stops. About " + q + " seconds per wall. The turn ends facing the north wall exactly where it began.");
  let order = [["north", "ne", "east"], ["east", "se", "south"], ["south", "sw", "west"], ["west", "nw", "north"]];
  let nth = {east: "a quarter turn", south: "a half turn, now facing the wall that was behind the camera", west: "three quarters of the turn"};
  order.forEach(function (step) {
    let c = passCorner(step[1], slots);
    if (c) parts.push(c);
    if (step[2] === "north") parts.push("The full turn completes and the north wall returns to the centre. " + fillsLine(slots.north) + ".");
    else parts.push("At " + nth[step[2]] + ", the " + step[2] + " wall fills the frame. " + fillsLine(slots[step[2]]) + ".");
  });
  parts.push(lockLine());
  return parts.join(" ");

}
function describeAngle(face, style, place, slots, where){
  let parts = [];
  parts.push(lead(style, place, "a wide shot from the exact centre of the room, at eye height, facing " + face + ". The camera holds a static shot."));
  parts.push(viewLine(face, slots));
  let mid = middleLevel(slots.middle, where);
  if (mid) parts.push(mid);
  parts.push(lockLine());
  return parts.join(" ");

}
function describeVertical(dir, style, place, slots, where){
  let motion = dir === "up"
    ? "a wide shot from the exact centre of the room. The camera tilts up with medium amplitude at slow speed and holds on the ceiling."
    : "a wide shot from the exact centre of the room. The camera tilts down with medium amplitude at slow speed and holds on the floor.";
  return lead(style, place, motion) + " " + middleVertical(slots.middle, where, dir) + " " + lockLine();

}
function buildOrbit(kind, dur, style, place, light, sound, north, east, south, west, ne, nw, se, sw, middle, where){
  let slots = slotMap(north, east, south, west, ne, nw, se, sw, middle);
  let k = String(kind || "sweep");
  let desc = "";
  if (k === "up" || k === "down") desc = describeVertical(k, style, place, slots, where);
  else if (k === "north" || k === "east" || k === "south" || k === "west") desc = describeAngle(k, style, place, slots, where);
  else desc = describeSweep(dur, style, place, slots, where);
  return "integrated_multimodal_description: " + desc + "\n\noverall_soundscape: " + soundBed(sound, light) + "\n\nnon_diegetic_music: N/A";

}
function surveyNotes(north, east, south, west, ne, nw, se, sw, middle){
  let rows = [
    ["North wall", north],
    ["East wall", east],
    ["South wall", south],
    ["West wall", west],
    ["Northeast corner", ne],
    ["Northwest corner", nw],
    ["Southeast corner", se],
    ["Southwest corner", sw],
    ["Middle", middle]
  ];
  let notes = [];
  let seen = {};
  rows.forEach(function (row) {
    let name = clean(row[1]);
    if (!name) return;
    let hasCue = false;
    name.toLowerCase().split(/[^a-z]+/).forEach(function (word) {
      if (cueWord(word)) hasCue = true;
    });
    if (!hasCue) notes.push(row[0] + ": \"" + name + "\" needs its own color or material, so a later scene prompt can point at it and not another object.");
    let noun = headNoun(name);
    if (!noun) return;
    if (seen[noun]) notes.push("\"" + seen[noun] + "\" and \"" + name + "\" are both a " + noun + ". Give each a different color, material, and shape.");
    else seen[noun] = name;
  });
  return notes.join("\n");

}
return {title, clipSeconds, clean, cap, nameCore, asSubject, sitLine, fillsLine, cueWord, headNoun, leftOf, rightOf, cornerKey, wallPhrase, slotMap, viewLine, middleLevel, middleVertical, passCorner, styleOpen, placeSentence, lockLine, soundBed, lead, describeSweep, describeAngle, describeVertical, buildOrbit, surveyNotes};
})();


const PC_CHAR = (function(){
function PN(name,kids){var n=Object.create(PN.prototype);n.getName=name;n.evaluateItem=name;n.kids=kids.map(function(k){return typeof k==="string"?PN(k,[]):PN(k[0],k[1]);});n.selectAll=n.kids;n.kids.forEach(function(k){if(k.kids.length&&!(k.getName in n))n[k.getName]=k;});return n;}
PN.prototype.toString=function(){if(!this.kids.length)return this.getName;var k=this.kids[Math.floor(Math.random()*this.kids.length)];return String(k);};
PN.prototype.valueOf=PN.prototype.toString;
PN.prototype.selectOne=function(){return this.kids[Math.floor(Math.random()*this.kids.length)];};

var title = "Character Generator";
var seeds = PN("seeds", [["drama", ["a night-shift nurse who keeps a notebook of every patient she lost", "a retired boxer running a failing gym in a coastal town", "a schoolteacher hiding that she cannot read music while leading the choir", "an eldest son who came home to bury a father he never forgave", "a wedding photographer who has never been in love"]], ["thriller", ["a courier who realises the package she carries is ticking", "an accountant who found one number that should not exist", "a hotel concierge who knows every guest's secret", "a hacker who owes money to the wrong family", "a bodyguard protecting the man who killed her brother"]], ["noir", ["a tired detective who only works cases nobody pays for", "a nightclub singer who sees everything from the stage", "a forger with one last commission", "a cab driver who keeps the city's secrets in his rear-view mirror", "a widow who is not grieving"]], ["horror", ["a lighthouse keeper who hears his name in the fog", "a child who draws things before they happen", "a caretaker of an empty sanatorium who still cooks for thirty", "a priest who lost his faith the night of the exorcism", "a podcaster who went into the woods for an episode and came back wrong"]], ["scifi", ["a salvage pilot who found a living crew on a derelict ship", "an android nanny who is about to be decommissioned", "a terraforming engineer who misses rain", "the last human translator for an alien embassy", "a clone who discovers she is number seven"]], ["fantasy", ["a disgraced royal cartographer who maps places that do not exist yet", "a dragon-tamer with a fear of heights", "a witch who runs a bakery and curses only bad tippers", "an aging knight who never won a single battle", "a thief who stole a god's name"]], ["comedy", ["a wedding planner who is terrible at love", "a motivational speaker having the worst week of his life", "a cat-sitter for the ultra-rich", "an overconfident magician whose tricks keep working for real", "a grandmother who becomes a viral streamer"]], ["romance", ["a florist and the man who orders flowers for a different woman every week", "two rival food-truck owners parked on the same corner", "a translator falling for the voice on the other end of the line", "a lifeguard who cannot swim", "a widower writing letters to his late wife and mailing them to the wrong address"]], ["action", ["a stunt double forced to do the real thing", "an ex-special-forces medic working as a school nurse", "a getaway driver with a strict no-guns rule", "a martial-arts instructor protecting her small-town dojo", "a bounty hunter whose target is her younger self"]], ["period", ["a Victorian lady botanist smuggling seeds out of an empire", "a 1920s jazz trumpeter in a segregated city", "a samurai turned tea master", "a 1960s NASA computer whose numbers nobody believes", "a Mughal court painter ordered to paint the emperor's enemy"]], ["western", ["a frontier dentist with a faster draw than the sheriff", "a widow running the only saloon in a dying town", "a Pinkerton agent hunting the outlaw who raised him", "a stagecoach driver who never lost a passenger", "a preacher with a price on his head"]], ["bollywood", ["a Mumbai dabbawala who delivers a love letter by mistake", "a small-town girl who wants to become a playback singer", "an angry young mill worker in 1970s Bombay", "a wedding choreographer with a broken heart", "a cricket-obsessed son of a strict police commissioner"]], ["anime", ["a shy high-school archer who can see spirits", "a delivery girl on a flying scooter in a floating city", "a mecha pilot who hates fighting", "a ramen chef with a legendary broth and a dark past", "a fox spirit pretending to be a transfer student"]], ["cartoon", ["a grumpy garden gnome who wants to see the ocean", "a tiny robot vacuum with big dreams", "an anxious ghost who is afraid of people", "a pigeon who thinks he is a secret agent", "a cheerful volcano who just wants friends"]], ["musicVideo", ["a lonely roller-skater at an empty 1980s rink", "a street dancer on the last night before the neighbourhood is demolished", "a singer driving across the desert at dawn", "a heartbroken DJ playing to an empty club", "a girl in a red coat walking through a city that freezes as she passes"]], ["commercial", ["a busy young mother with five minutes of calm", "a weekend cyclist chasing a personal record", "a grandfather learning to video-call his grandchildren", "a barista who knows every regular's order", "a startup founder pitching at sunrise"]], ["documentary", ["a beekeeper on the last organic farm in the valley", "a street-food vendor who has worked the same corner for forty years", "a deep-sea welder on an oil rig", "a teenage chess prodigy from a refugee camp", "a night-train conductor on his final route"]]]);
return {title, seeds};
})();

const PC_CHARUI = (function(){
var GENRES = [["drama", "Drama (live-action)"], ["thriller", "Thriller"], ["noir", "Noir"], ["horror", "Horror"], ["scifi", "Sci-fi"], ["fantasy", "Fantasy"], ["comedy", "Comedy"], ["romance", "Romance"], ["action", "Action"], ["period", "Period / historical"], ["western", "Western"], ["bollywood", "Bollywood"], ["anime", "Anime"], ["cartoon", "Cartoon / 3D animation"], ["musicVideo", "Music video"], ["commercial", "TV commercial"], ["documentary", "Documentary"]];
function genreLabel(g){ for (var i=0;i<GENRES.length;i++) if (GENRES[i][0]===g) return GENRES[i][1]; return g || "Drama"; }
  var FIELDS = [
    { k: "age", label: "Age", ph: "e.g. late 30s" },
    { k: "look", label: "Ethnicity / look", ph: "heritage, skin tone, overall look" },
    { k: "build", label: "Build", ph: "height, body type" },
    { k: "hair", label: "Hair", ph: "color, length, style, facial hair" },
    { k: "face", label: "Face", ph: "eyes, brows, nose, lips, marks, scars", long: true },
    { k: "outfit", label: "Outfit", ph: "every garment with color and material, shoes, accessories", long: true },
    { k: "personality", label: "Personality", ph: "core traits and how they show", long: true },
    { k: "backstory", label: "Backstory", ph: "origin, defining event, what they do now", long: true },
    { k: "arc", label: "Arc", ph: "where they start, what changes them, where they end", long: true },
    { k: "voice", label: "Voice / speech style", ph: "timbre, accent, pace, verbal habits" },
    { k: "quirks", label: "Quirks", ph: "habits and tics you can see on camera" },
    { k: "goals", label: "Goals / fears", ph: "want, need, fear" },
    { k: "relationships", label: "Relationships", ph: "key people and the dynamic" }
  ];
  var FKEYS = FIELDS.map(function (f) { return f.k; });
  var FILLABLE = ["name", "role"].concat(FKEYS);
  var VISUAL_KEYS = ["age", "look", "build", "face", "hair", "outfit"];
  var MULTILINE = { backstory: true, arc: true, relationships: true, personality: true };

  var KEYDOC = {
    name: "a fitting full name, or a strong label if they are nameless",
    role: "their role or archetype in the story, a few words",
    age: "apparent age, e.g. early 30s",
    look: "ethnicity or heritage, skin tone, overall look",
    build: "height and body type",
    face: "face shape, eyes, brows, nose, lips, skin details, distinguishing marks",
    hair: "color, length, texture, style, facial hair if any",
    outfit: "signature outfit: every garment with color and material, footwear, accessories",
    personality: "3 to 6 core traits and how they show in behaviour",
    backstory: "3 to 5 sentences: origin, a defining event or wound, what they do now",
    arc: "where they start, what changes them, where they end up",
    voice: "voice timbre, pitch, pace, accent, speech patterns, a typical phrase",
    quirks: "2 to 4 specific habits or tics that are visible on camera",
    goals: "what they want, what they really need, what they fear",
    relationships: "2 to 4 key relationships as Name (relation): dynamic, separated by semicolons"
  };

  var ALIASES = {
    "name": "name", "full name": "name", "character name": "name",
    "role": "role", "archetype": "role", "role archetype": "role", "role and archetype": "role",
    "age": "age", "apparent age": "age",
    "look": "look", "ethnicity": "look", "ethnicity look": "look", "ethnicity and look": "look", "heritage": "look", "appearance": "look", "skin": "look", "skin tone": "look",
    "build": "build", "body": "build", "body type": "build", "physique": "build", "height and build": "build", "height build": "build",
    "face": "face", "facial features": "face", "features": "face",
    "hair": "hair", "hairstyle": "hair", "hair style": "hair",
    "outfit": "outfit", "costume": "outfit", "clothing": "outfit", "clothes": "outfit", "wardrobe": "outfit", "signature outfit": "outfit",
    "personality": "personality", "traits": "personality", "personality traits": "personality",
    "backstory": "backstory", "back story": "backstory", "background": "backstory", "history": "backstory",
    "arc": "arc", "character arc": "arc",
    "voice": "voice", "voice speech style": "voice", "voice and speech style": "voice", "speech": "voice", "speech style": "voice", "voice and speech": "voice",
    "quirks": "quirks", "quirk": "quirks", "habits": "quirks", "tics": "quirks", "quirks and habits": "quirks",
    "goals": "goals", "goals fears": "goals", "goals and fears": "goals", "goal": "goals", "fears": "goals", "wants": "goals", "motivation": "goals",
    "relationships": "relationships", "relationship": "relationships", "relations": "relationships"
  };
  var OUT_ALIASES = {
    "visual": "visual", "visual line": "visual", "visual description": "visual", "appearance": "visual", "look": "visual",
    "voice": "voice", "voice description": "voice", "voice line": "voice"
  };

  var STYLE = {
    anime: "anime character design sheet, clean cel-shaded linework, flat colors",
    cartoon: "stylized 3D animated character design, appealing shapes, clean materials",
    period: "photorealistic live-action costume reference, period-accurate details",
    western: "photorealistic live-action costume reference, frontier-era details",
    fantasy: "photorealistic live-action fantasy costume reference",
    scifi: "photorealistic live-action sci-fi costume reference"
  };

  function normLabel(s) { return String(s || "").toLowerCase().replace(/[^a-z]+/g, " ").trim(); }

  // Lenient "key: value" parser. Unknown lines continue the previous key. Code fences, markdown and numbering are ignored.
  function parseFields(raw, allowed, aliases) {
    aliases = aliases || ALIASES;
    var allow = {};
    (allowed || []).forEach(function (k) { allow[k] = true; });
    var out = {};
    var cur = null;
    var lines = String(raw || "").replace(/\r/g, "").split("\n");
    for (var i = 0; i < lines.length; i++) {
      var t = lines[i].trim();
      if (/^```/.test(t)) continue;
      if (/^(END|<\/?end>)\.?$/i.test(t)) break;
      t = t.replace(/\*\*|__/g, "");
      for (var j = 0; j < 3; j++) t = t.replace(/^\s*(?:[-*\u2022+>#]+|\(?\d{1,2}[.)\]]|\[\d{1,2}\])\s*/, "");
      t = t.trim();
      if (!t) { if (cur && MULTILINE[cur] && out[cur] && !/\n$/.test(out[cur])) out[cur] += "\n\n"; continue; }
      var m = /^([A-Za-z][A-Za-z \/&()'-]{0,40}?)\s*[:=]\s*(.*)$/.exec(t);
      var key = m ? aliases[normLabel(m[1])] : null;
      if (key) {
        if (allow[key] && out[key] === undefined) { cur = key; out[key] = m[2].replace(/`{3,}\w*/g, "").trim(); }
        else cur = null;
        continue;
      }
      if (cur) {
        var sep = MULTILINE[cur] && /\n$/.test(out[cur]) ? "" : (out[cur] ? " " : "");
        out[cur] += sep + t;
      }
    }
    Object.keys(out).forEach(function (k) {
      var v = out[k].replace(/\n{3,}/g, "\n\n").trim().replace(/^["'\u201c]+|["'\u201d]+$/g, "").trim();
      if (!v || /^(n\/a|none|tbd|\.\.\.|-)$/i.test(v) || /^(sorry|i'm sorry|i am sorry|i can't|i cannot|i won't|as an ai)\b/i.test(v)) delete out[k];
      else out[k] = v;
    });
    return out;
  }

  function contextText(d, skip) {
    skip = skip || {};
    var lines = [];
    if (!skip.name && d.name.trim()) lines.push("name: " + d.name.trim());
    if (!skip.role && d.role.trim()) lines.push("role: " + d.role.trim());
    lines.push("genre: " + genreLabel(d.genre));
    if (d.seed.trim()) lines.push("seed idea: " + d.seed.trim().replace(/\s+/g, " "));
    FKEYS.forEach(function (k) {
      if (!skip[k] && String(d.f[k]).trim()) lines.push(k + ": " + String(d.f[k]).trim().replace(/\s*\n\s*/g, " / "));
    });
    return lines.join("\n");
  }
  var PIPE = "You are a character designer for an AI filmmaker. The character will become a Krea2 / Qwen Image Edit character turnaround sheet, then first frames, then MiniMax video, so every visual detail must be concrete, specific and visible on camera, and consistent with the genre.";
  var PLAIN = "Plain text only: no markdown, no bold, no bullets, no numbering, no code fences, no intro, no notes. Any subject is allowed; explicit content only if the user wrote it.";

  function developPrompt(d, keys) {
    return PIPE + "\n\nCHARACTER SO FAR (written by the user, binding: do not contradict or change any of it, build on it):\n" + contextText(d) +
      "\n\nWrite ONLY these missing fields, one per line, in exactly this format:\nkey: value\n\nKeys to write, in this order:\n" +
      keys.map(function (k) { return "- " + k + ": " + KEYDOC[k]; }).join("\n") +
      "\n\nKeys are lowercase and spelled exactly as shown. Each value stays on one line. " + PLAIN + " After the last key write a line with only END.";
  }
  function regenPrompt(d, k) {
    var skip = {}; skip[k] = true;
    var cur = k === "name" ? d.name : (k === "role" ? d.role : d.f[k]);
    return PIPE + "\n\nCHARACTER (binding, keep consistent with it):\n" + contextText(d, skip) +
      "\n\nWrite a fresh alternative for ONE field only: " + k + " (" + KEYDOC[k] + ")." +
      (String(cur || "").trim() ? "\nCurrent value, to be replaced with something different but still consistent: " + String(cur).trim().replace(/\s+/g, " ") : "") +
      "\n\nOutput exactly one line:\n" + k + ": value\n" + PLAIN + " Then a line with only END.";
  }
  function deepenPrompt(d) {
    return PIPE + "\n\nCHARACTER (binding):\n" + contextText(d, { backstory: true, arc: true }) +
      "\n\nCURRENT BACKSTORY (written so far; keep every fact in it):\n" + (d.f.backstory.trim() || "(empty)") +
      "\n\nCURRENT ARC (keep every fact in it):\n" + (d.f.arc.trim() || "(empty)") +
      "\n\nDeepen the story. Output exactly two fields:\nbackstory: 2 to 4 short paragraphs. Keep every existing fact, then add specific formative events with places and years or ages, a secret they hide, a contradiction in them, and what they did to survive.\narc: 3 to 5 sentences: the lie they believe, the inciting pressure, the turning point, the cost, and who they are at the end.\n\nStart each field with its key and a colon. Paragraphs may be separated by blank lines. " + PLAIN + " After the arc write a line with only END.";
  }
  function outputsPrompt(d) {
    return PIPE + "\n\nCHARACTER:\n" + contextText(d) +
      "\n\nWrite two fields, one line each:\nvisual: ONE dense English sentence (at most 70 words) with only fixed visible traits: apparent age, gender presentation, build, skin tone and ethnicity, hair, face and distinguishing features, then the full outfit with colors and materials, footwear and accessories. No name, no personality, no actions, no emotions, no camera or background words.\nvoice: one or two sentences describing the voice for a voice model: gender presentation, age, timbre, pitch, pace, accent, delivery and emotional color. No name.\n\n" + PLAIN + " After the voice line write a line with only END.";
  }

  function fallbackVisual(d) {
    var f = d.f, parts = [];
    var who = [f.age, f.look].map(function (x) { return String(x || "").trim(); }).filter(Boolean).join(", ");
    if (who) parts.push(who);
    if (f.build.trim()) parts.push(f.build.trim() + (/build/i.test(f.build) ? "" : " build"));
    if (f.hair.trim()) parts.push(f.hair.trim());
    if (f.face.trim()) parts.push(f.face.trim());
    var s = parts.join(", ").replace(/\s+/g, " ");
    if (f.outfit.trim()) s += (s ? ", wearing " : "wearing ") + f.outfit.trim().replace(/\s+/g, " ");
    s = s.replace(/[.\s]+$/, "");
    return s ? s.charAt(0).toUpperCase() + s.slice(1) + "." : "";
  }
  function buildSheet(name, visual, genre, refMode) {
    var v = String(visual || "").trim().replace(/[.\s]+$/, "");
    var style = STYLE[genre] || "photorealistic live-action character reference, natural skin texture, cinematic realism";
    var who = String(name || "").trim();
    var head = refMode
      ? "Using the person in the reference image, keep their face, identity, skin tone and hair exactly the same, and create a character turnaround reference sheet" + (who ? " of " + who : "") + (v ? ": " + v : "") + "."
      : "Character turnaround reference sheet" + (who ? " of " + who : "") + (v ? ": " + v : "") + ".";
    return head + " " + style.charAt(0).toUpperCase() + style.slice(1) + ".\n" +
      "Top row, full body, head to toe, evenly spaced and the same scale, left to right: front view, three-quarter view, side profile view, back view. Neutral standing pose, arms relaxed slightly away from the body, feet visible.\n" +
      "Bottom row, head-and-shoulders close-ups of the same face: neutral, smiling, angry, sad, surprised, determined.\n" +
      "Same character in every view: identical face, hairstyle, body proportions, outfit, colors and accessories.\n" +
      "Plain pure white background, even soft studio lighting, no cast shadows, no props, no scenery, no text, no labels, no watermark, sharp focus, high detail.";
  }

return {GENRES, FIELDS, FKEYS, FILLABLE, parseFields, developPrompt, regenPrompt, deepenPrompt, outputsPrompt, fallbackVisual, buildSheet, OUT_ALIASES};
})();

// ===== end Perchance logic =====
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const TABS = ['ocr','translate','prompt','roleplay','rewrite','chat','caption','extract','t2va','ref2va','sound','orbit','character'];
let CFG = {defaults:{}, systems:{}, languages:[], settings:{}};
let MODELS = [];
let MODEL_CAPS = {};
const IMAGE_TABS = new Set(['ocr', 'caption', 'extract']);
const ST = {};

function S(tab){ CFG.settings[tab] = CFG.settings[tab] || {}; return CFG.settings[tab]; }
let saveT = null;
function persist(){
  localStorage.setItem('wiz-studio-settings', JSON.stringify(CFG.settings));
  clearTimeout(saveT);
  saveT = setTimeout(() => fetch('/api/settings', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify(CFG.settings)}).catch(()=>{}), 400);
}
function currentTab(){ return S('_ui').tab || 'ocr'; }

function applyTheme(){
  const g = S('_ui');
  if (g.theme) document.documentElement.dataset.theme = g.theme; else delete document.documentElement.dataset.theme;
  if (g.accent) document.documentElement.dataset.accent = g.accent; else delete document.documentElement.dataset.accent;
  $('#theme-sel').value = g.theme || '';
  $('#accent-sel').value = g.accent || '';
}
$('#theme-sel').onchange = e => { S('_ui').theme = e.target.value; applyTheme(); persist(); };
$('#accent-sel').onchange = e => { S('_ui').accent = e.target.value; applyTheme(); persist(); };

function showTab(t){
  if (!TABS.includes(t)) t = 'ocr';
  $$('nav.tabs button').forEach(b => b.classList.toggle('active', b.dataset.tab === t));
  $$('section.tab').forEach(s => s.classList.toggle('active', s.id === 'tab-' + t));
  S('_ui').tab = t;
  persist();
  if (location.hash !== '#' + t) history.replaceState(null, '', '#' + t);
}
$$('nav.tabs button').forEach(b => b.onclick = () => showTab(b.dataset.tab));
window.addEventListener('hashchange', () => {
  const t = location.hash.slice(1);
  if (TABS.includes(t) && t !== currentTab()) showTab(t);
});

function baseName(m){ return String(m || '').split(':')[0]; }
function capsOf(name){
  if (!name) return null;
  if (MODEL_CAPS[name]) return MODEL_CAPS[name];
  const hit = Object.keys(MODEL_CAPS).find(k => baseName(k) === baseName(name));
  return hit ? MODEL_CAPS[hit] : null;
}
function isVision(name){ const c = capsOf(name); return !!(c && c.includes('vision')); }
function modelsFor(tab){
  if (!IMAGE_TABS.has(tab) || !MODELS.length) return MODELS;
  const vis = MODELS.filter(isVision);
  return vis.length ? vis : MODELS;
}
function matchInstalled(pool, want){
  if (!want || !pool.length) return '';
  return pool.find(m => m === want) || pool.find(m => baseName(m) === baseName(want)) || '';
}
function pickModel(tab, wanted){
  const want = wanted || S(tab).model || (CFG.defaults || {})[tab];
  const pool = modelsFor(tab);
  if (!pool.length) return want || '';
  return matchInstalled(pool, want) || matchInstalled(pool, (CFG.defaults || {})[tab]) || pool[0];
}
function modelLabel(m){
  const c = capsOf(m);
  if (!c) return m;
  return m + (c.includes('vision') ? '  ·  vision' : '  ·  text');
}
function buildModelBars(){
  $$('.modelbar').forEach(bar => {
    const tab = bar.dataset.for, s = S(tab);
    bar.innerHTML = `
      <label>Model <select class="mb-model"></select></label>
      <label>Temperature <span class="mb-tv"></span><input type="range" class="mb-temp" min="0" max="2" step="0.05"></label>
      <label class="inline"><input type="checkbox" class="mb-think"> hide &lt;think&gt;</label>
      <details style="flex-basis:100%"><summary>Advanced · system prompt, context</summary>
        <textarea class="mb-sys" placeholder="(empty = no system message)"></textarea>
        <div class="bar" style="margin-top:6px">
          <label>Context tokens <input class="mb-ctx" type="number" min="0" max="131072" step="1024" placeholder="model default"></label>
          <button type="button" class="mb-reset">Reset system prompt</button>
          <span class="hint">0 keeps the model's own context.</span>
        </div>
      </details>`;
    const temp = $('.mb-temp', bar), tv = $('.mb-tv', bar), think = $('.mb-think', bar), sys = $('.mb-sys', bar), ctx = $('.mb-ctx', bar);
    temp.value = s.temperature ?? (tab === 'roleplay' ? 0.9 : (tab === 'translate' || tab === 'ocr' || tab === 'caption' || tab === 'extract') ? 0.1 : 0.7);
    tv.textContent = temp.value;
    temp.oninput = () => { tv.textContent = temp.value; s.temperature = +temp.value; persist(); };
    think.checked = s.hideThink ?? true;
    think.onchange = () => { s.hideThink = think.checked; persist(); };
    sys.value = s.system ?? (CFG.systems || {})[tab] ?? '';
    sys.oninput = () => { s.system = sys.value; persist(); };
    ctx.value = s.numCtx || '';
    ctx.oninput = () => { s.numCtx = Math.max(0, Math.min(131072, +ctx.value || 0)); persist(); };
    $('.mb-reset', bar).onclick = () => { sys.value = (CFG.systems || {})[tab] || ''; delete s.system; persist(); };
    $('.mb-model', bar).onchange = e => { s.model = e.target.value; persist(); };
  });
  fillModels();
}
function fillModels(){
  $$('.modelbar').forEach(bar => {
    const tab = bar.dataset.for, sel = $('.mb-model', bar);
    const saved = S(tab).model || '';
    const cur = pickModel(tab);
    const list = modelsFor(tab);
    const shown = list.length ? list : (cur ? [cur] : []);
    sel.innerHTML = shown.map(m => `<option value="${esc(m)}">${esc(modelLabel(m))}</option>`).join('');
    if (cur) sel.value = cur;
    const savedCaps = capsOf(saved);
    if (IMAGE_TABS.has(tab) && saved && cur && savedCaps && !savedCaps.includes('vision') && baseName(saved) !== baseName(cur)) {
      S(tab).model = cur;
      persist();
      if (ST[tab] && ST[tab].status) ST[tab].status.textContent = baseName(saved) + ' cannot read images. Using ' + baseName(cur) + '.';
    }
  });
}
function modelOf(tab){ return $(`.modelbar[data-for="${tab}"] .mb-model`).value; }
function tempOf(tab){ return +$(`.modelbar[data-for="${tab}"] .mb-temp`).value; }
function sysOf(tab){ return $(`.modelbar[data-for="${tab}"] .mb-sys`).value.trim(); }
function hideThink(tab){ return $(`.modelbar[data-for="${tab}"] .mb-think`).checked; }
function ctxOf(tab){ const n = +($(`.modelbar[data-for="${tab}"] .mb-ctx`).value || 0); return n > 0 ? Math.round(n) : 0; }
function setModel(tab, name){
  const sel = $(`.modelbar[data-for="${tab}"] .mb-model`);
  const hit = [...sel.options].find(o => o.value === name || o.value.split(':')[0] === String(name).split(':')[0]);
  if (hit) { sel.value = hit.value; S(tab).model = hit.value; persist(); return true; }
  if (MODELS.length && ST[tab]) ST[tab].status.textContent = name + ' is not installed.';
  return false;
}
async function loadModels(){
  const st = $('#ollama-state');
  try {
    const r = await (await fetch('/api/models')).json();
    MODELS = r.models || [];
    MODEL_CAPS = r.capabilities || {};
    if (r.ok) { st.textContent = 'Ollama · ' + MODELS.length + ' models'; st.className = 'pill ok'; banner(''); }
    else { st.textContent = 'Ollama offline'; st.className = 'pill bad'; banner(r.error + ' — start it from AI Dock or run ollama serve, then click Models.'); }
  } catch (e) { st.textContent = 'Studio server unreachable'; st.className = 'pill bad'; }
  fillModels();
}
$('#refresh-models').onclick = loadModels;
function banner(t){ const b = $('#banner'); b.textContent = t; b.classList.toggle('show', !!t); }

let toastT = null;
function toast(t){ const el = $('#toast'); el.textContent = t; el.classList.add('show'); clearTimeout(toastT); toastT = setTimeout(() => el.classList.remove('show'), 2200); }
function counts(t){ const s = String(t || ''); const words = s.trim() ? s.trim().split(/\s+/).length : 0; return words + ' words · ' + s.length + ' chars'; }
function esc(s){ return String(s).replace(/[&<>"]/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;'}[c])); }
function stripThink(t){ return t.replace(/<think>[\s\S]*?(<\/think>|$)/g, '').replace(/^\s+/, ''); }
function stamp(){ return new Date().toISOString().slice(0,19).replace(/[:T]/g,'-'); }
function download(name, text){ const a = document.createElement('a'); a.href = URL.createObjectURL(new Blob([text], {type:'text/plain'})); a.download = name; a.click(); setTimeout(() => URL.revokeObjectURL(a.href), 2000); }
function downloadB64(name, b64, mime){
  const bin = atob(b64), bytes = new Uint8Array(bin.length);
  for (let i = 0; i < bin.length; i++) bytes[i] = bin.charCodeAt(i);
  const a = document.createElement('a');
  a.href = URL.createObjectURL(new Blob([bytes], {type: mime || 'application/octet-stream'}));
  a.download = name; a.click();
  setTimeout(() => URL.revokeObjectURL(a.href), 2000);
}
async function copyText(t, st){
  try { await navigator.clipboard.writeText(t || ''); }
  catch (e) { const ta = document.createElement('textarea'); ta.value = t || ''; document.body.appendChild(ta); ta.select(); document.execCommand('copy'); ta.remove(); }
  if (st) { st.textContent = 'Copied.'; st.classList.remove('err'); }
  toast('Copied');
}
function miniMd(src){
  const lines = String(src || '').split('\n');
  let html = '', inList = false, inCode = false, code = [];
  const closeList = () => { if (inList) { html += '</ul>'; inList = false; } };
  const inline = (s) => esc(s).replace(/`([^`]+)`/g, '<code>$1</code>').replace(/\*\*([^*]+)\*\*/g, '<strong>$1</strong>').replace(/(^|[^*])\*([^*\n]+)\*/g, '$1<em>$2</em>');
  for (const line of lines) {
    if (line.trim().startsWith('```')) {
      if (inCode) { closeList(); html += '<pre><code>' + esc(code.join('\n')) + '</code></pre>'; code = []; inCode = false; }
      else { closeList(); inCode = true; }
      continue;
    }
    if (inCode) { code.push(line); continue; }
    if (/^\s*[-*]\s+/.test(line)) {
      if (!inList) { html += '<ul>'; inList = true; }
      html += '<li>' + inline(line.replace(/^\s*[-*]\s+/, '')) + '</li>';
      continue;
    }
    closeList();
    const h = /^(#{1,3})\s+(.*)$/.exec(line);
    if (h) { html += '<h' + h[1].length + '>' + inline(h[2]) + '</h' + h[1].length + '>'; continue; }
    if (line.includes('|')) { html += '<pre>' + esc(line) + '</pre>'; continue; }
    if (!line.trim()) continue;
    html += '<p>' + inline(line) + '</p>';
  }
  closeList();
  if (inCode) html += '<pre><code>' + esc(code.join('\n')) + '</code></pre>';
  return html;
}
function wordDiffHtml(a, b){
  const tok = (s) => String(s || '').split(/(\s+)/).filter(x => x.length);
  const A = tok(a), B = tok(b);
  if (a === b) return '<p class="hint">No changes.</p>';
  if (A.length > 700 || B.length > 700) return '<p class="hint">This text is long, so the change view stays off.</p>';
  const n = A.length, m = B.length;
  const dp = Array.from({length: n + 1}, () => new Uint16Array(m + 1));
  for (let i = n - 1; i >= 0; i--) for (let j = m - 1; j >= 0; j--) dp[i][j] = A[i] === B[j] ? dp[i + 1][j + 1] + 1 : Math.max(dp[i + 1][j], dp[i][j + 1]);
  let i = 0, j = 0, out = '';
  while (i < n && j < m) {
    if (A[i] === B[j]) { out += esc(A[i]); i++; j++; }
    else if (dp[i + 1][j] >= dp[i][j + 1]) { out += '<del>' + esc(A[i]) + '</del>'; i++; }
    else { out += '<ins>' + esc(B[j]) + '</ins>'; j++; }
  }
  while (i < n) out += '<del>' + esc(A[i++]) + '</del>';
  while (j < m) out += '<ins>' + esc(B[j++]) + '</ins>';
  return out || '<p class="hint">No changes.</p>';
}

function paintOut(tab, text){
  const st = ST[tab];
  if (!st || !st.out) return;
  st.out.value = text;
  st.out.scrollTop = st.out.scrollHeight;
  if (st.count) st.count.textContent = counts(text);
  if (st.previewOn && st.md) st.md.innerHTML = miniMd(text);
}
function buildOutputs(){
  $$('[data-output]').forEach(box => {
    const tab = box.dataset.output;
    box.innerHTML = `<div class="out-tools">
        <span class="count"></span><span style="flex:1"></span>
        <button type="button" class="o-preview">Preview</button>
        <button type="button" class="o-lib">Library</button>
        <button type="button" class="o-stop">Stop</button>
        <button type="button" class="o-copy">Copy</button>
        <button type="button" class="o-save">Save</button>
        <label class="inline sendto">Send to <select class="o-send">
          <option value="">…</option>
          <option value="rewrite">Rewrite</option>
          <option value="translate">Translate</option>
          <option value="prompt">Prompt</option>
          <option value="chat">Chat</option>
          <option value="t2va">T2VA direction</option>
          <option value="sound">Soundscape scene</option>
          <option value="character">Character seed</option>
        </select></label>
      </div>
      <textarea class="out"></textarea>
      <div class="mdview" hidden></div>
      <div class="status"></div>`;
    const out = $('.out', box), md = $('.mdview', box);
    ST[tab] = {out, status: $('.status', box), ctrl: null, count: $('.count', box), md, previewOn: false, label: '', last: ''};
    const refresh = () => { ST[tab].count.textContent = counts(out.value); if (ST[tab].previewOn) md.innerHTML = miniMd(out.value); };
    out.addEventListener('input', refresh);
    $('.o-stop', box).onclick = () => stop(tab);
    $('.o-copy', box).onclick = () => copyText(out.value, ST[tab].status);
    $('.o-save', box).onclick = () => download(tab + '-' + stamp() + '.' + (tab === 'ocr' ? 'md' : 'txt'), out.value);
    $('.o-lib', box).onclick = () => openLib(tab);
    $('.o-preview', box).onclick = () => {
      ST[tab].previewOn = !ST[tab].previewOn;
      md.hidden = !ST[tab].previewOn;
      out.hidden = ST[tab].previewOn;
      $('.o-preview', box).classList.toggle('primary', ST[tab].previewOn);
      refresh();
    };
    $('.o-send', box).onchange = (e) => { const dest = e.target.value; e.target.value = ''; if (dest) sendTo(dest, out.value); };
    refresh();
  });
}
function ensureRuntime(){
  const keep = (tab, status) => { ST[tab] = Object.assign({ctrl:null, out:null, label:'', last:''}, ST[tab], {status}); };
  keep('roleplay', $('#rp-status'));
  keep('chat', $('#ch-status'));
  keep('caption', $('#cap-status'));
}
function stop(tab){ const s = ST[tab]; if (s && s.ctrl) { s.ctrl.abort(); s.ctrl = null; } }
async function busy(btn, fn){ if (!btn || btn.disabled) return; btn.disabled = true; try { await fn(); } finally { btn.disabled = false; } }
function sendTo(tab, text){
  if (!String(text || '').trim()) { toast('Nothing to send'); return; }
  if (tab === 'rewrite') { showTab('rewrite'); $('#rw-text').value = text; }
  else if (tab === 'translate') { showTab('translate'); $('#tr-text').value = text; }
  else if (tab === 'prompt') { showTab('prompt'); $('#pg-idea').value = text; }
  else if (tab === 't2va') { showTab('t2va'); setKeep('t2-dir', text); }
  else if (tab === 'sound') { showTab('sound'); setKeep('sd-text', text); }
  else if (tab === 'character') { showTab('character'); setKeep('cg-seed', text); }
  else if (tab === 'chat') { showTab('chat'); $('#ch-input').value = text; $('#ch-input').focus(); }
  toast('Sent to ' + tab);
}

async function streamChat(tab, messages, onText){
  stop(tab);
  const slot = ST[tab];
  if (!slot) return '';
  const ctrl = new AbortController();
  if (!ensureVisionModel(tab, messages)) return '';
  slot.ctrl = ctrl;
  slot.last = 'run';
  let full = '', t0 = performance.now(), evals = 0, evalNs = 0;
  const status = slot.status;
  const lead = () => slot.label ? slot.label + ' · ' : '';
  status.classList.remove('err');
  status.textContent = lead() + modelOf(tab) + ' is working…';
  try {
    const payload = {model: modelOf(tab), temperature: tempOf(tab), messages};
    const ctx = ctxOf(tab);
    if (ctx) payload.num_ctx = ctx;
    const res = await fetch('/api/chat', {method:'POST', headers:{'Content-Type':'application/json'}, signal: ctrl.signal, body: JSON.stringify(payload)});
    const rd = res.body.getReader(), dec = new TextDecoder();
    let buf = '';
    for (;;) {
      const {done, value} = await rd.read();
      if (done) break;
      buf += dec.decode(value, {stream:true});
      let nl;
      while ((nl = buf.indexOf('\n')) >= 0) {
        const line = buf.slice(0, nl).trim();
        buf = buf.slice(nl + 1);
        if (!line) continue;
        let j;
        try { j = JSON.parse(line); } catch (err) { continue; }
        if (j.error) throw new Error(j.error);
        if (j.message && j.message.content) { full += j.message.content; onText(hideThink(tab) ? stripThink(full) : full); status.textContent = lead() + 'Streaming…'; }
        if (j.done) { evals = j.eval_count || 0; evalNs = j.eval_duration || 0; }
      }
    }
    const sec = (performance.now() - t0) / 1000;
    let extra = '';
    if (evals) { const tps = evalNs ? evals / (evalNs / 1e9) : evals / Math.max(sec, 0.001); extra = ' · ' + evals + ' tokens · ' + tps.toFixed(1) + ' tok/s'; }
    status.textContent = lead() + 'Done in ' + sec.toFixed(1) + 's' + extra;
    slot.last = 'ok';
  } catch (e) {
    if (e.name === 'AbortError') { status.textContent = lead() + 'Stopped'; slot.last = 'abort'; }
    else { status.textContent = e.message; status.classList.add('err'); slot.last = 'err'; }
  } finally { if (slot.ctrl === ctrl) slot.ctrl = null; }
  return hideThink(tab) ? stripThink(full) : full;
}
function messagesNeedVision(messages){
  return (messages || []).some(m => Array.isArray(m.images) && m.images.length);
}
function ensureVisionModel(tab, messages){
  if (!messagesNeedVision(messages)) return true;
  const name = modelOf(tab);
  const caps = capsOf(name);
  if (!caps || caps.includes('vision')) return true;
  const pool = MODELS.filter(isVision);
  const fb = matchInstalled(pool, (CFG.defaults || {})[tab]) || matchInstalled(pool, (CFG.defaults || {}).extract) || pool[0] || '';
  const slot = ST[tab];
  if (!fb) {
    slot.last = 'err';
    slot.status.classList.add('err');
    slot.status.textContent = baseName(name) + ' cannot read images, and no vision model is installed.';
    return false;
  }
  setModel(tab, fb);
  toast(baseName(name) + ' cannot read images. Using ' + baseName(fb) + '.');
  return true;
}
function withSys(tab, msgs, sysOverride){ const s = sysOverride ?? sysOf(tab); return s ? [{role:'system', content:s}, ...msgs] : msgs; }
function runInto(tab, msgs){ paintOut(tab, ''); return streamChat(tab, msgs, t => paintOut(tab, t)); }
async function remember(tab, title, text, meta){
  const body = String(text || '').trim();
  if (!body) return;
  try {
    await fetch('/api/library', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({op:'add', item:{tab, title, text: body, meta: meta || {}}})});
  } catch (e) {}
}

function sniffMime(b64){
  if (String(b64).startsWith('/9j/')) return 'image/jpeg';
  if (String(b64).startsWith('iVBORw')) return 'image/png';
  if (String(b64).startsWith('UklGR')) return 'image/webp';
  return 'image/png';
}
function readB64(file){ return new Promise((ok, no) => { const r = new FileReader(); r.onload = () => ok(String(r.result).split(',')[1]); r.onerror = no; r.readAsDataURL(file); }); }
function shrinkImage(b64, max){
  max = max || 1600;
  return new Promise(resolve => {
    const img = new Image();
    img.onload = () => {
      const scale = Math.min(1, max / Math.max(img.width || 1, img.height || 1));
      if (scale === 1 && b64.length < 900000) { resolve(b64); return; }
      const c = document.createElement('canvas');
      c.width = Math.max(1, Math.round(img.width * scale));
      c.height = Math.max(1, Math.round(img.height * scale));
      c.getContext('2d').drawImage(img, 0, 0, c.width, c.height);
      resolve(c.toDataURL('image/jpeg', 0.86).split(',')[1]);
    };
    img.onerror = () => resolve(b64);
    img.src = 'data:' + sniffMime(b64) + ';base64,' + b64;
  });
}
function setupDrop(dropId, fileId, onFiles){
  const d = $('#' + dropId), f = $('#' + fileId);
  d.onclick = e => { if (e.target !== f) f.click(); };
  f.onchange = () => { onFiles([...f.files]); f.value = ''; };
  d.ondragover = e => { e.preventDefault(); d.classList.add('over'); };
  d.ondragleave = () => d.classList.remove('over');
  d.ondrop = e => { e.preventDefault(); d.classList.remove('over'); onFiles([...e.dataTransfer.files]); };
}
function drawThumbs(id, imgs, onRemove){
  const el = $('#' + id);
  el.innerHTML = '';
  imgs.forEach((b, i) => {
    const wrap = document.createElement('div');
    wrap.className = 'thumb';
    const img = document.createElement('img');
    img.alt = '';
    img.src = 'data:' + sniffMime(b) + ';base64,' + b;
    wrap.appendChild(img);
    if (onRemove) {
      const x = document.createElement('button');
      x.type = 'button'; x.className = 'thumb-x'; x.textContent = '×'; x.title = 'Remove';
      x.onclick = (ev) => { ev.stopPropagation(); onRemove(i); };
      wrap.appendChild(x);
    }
    el.appendChild(wrap);
  });
}
async function rememberPaste(files){
  const out = [];
  for (const f of files) {
    if (!f || (!/^image\//.test(f.type) && !/\.(png|jpe?g|webp)$/i.test(f.name || ''))) continue;
    out.push({name: f.name || 'image', b64: await shrinkImage(await readB64(f))});
  }
  return out;
}

let libTimer = null;
function openLib(tab){
  $('#lib-modal').classList.add('open');
  if (tab) $('#lib-filter').value = tab;
  $('#lib-q').focus();
  renderLib();
}
async function renderLib(){
  const q = $('#lib-q').value.trim();
  const tab = $('#lib-filter').value;
  let data;
  try { data = await (await fetch('/api/library?q=' + encodeURIComponent(q) + '&tab=' + encodeURIComponent(tab))).json(); }
  catch (e) { return; }
  const ul = $('#lib-list');
  ul.innerHTML = '';
  const items = data.items || [];
  if (!items.length) {
    const li = document.createElement('li');
    li.textContent = 'Nothing saved yet.';
    ul.appendChild(li);
    return;
  }
  items.forEach(it => {
    const li = document.createElement('li');
    const left = document.createElement('div');
    left.className = 'lib-hit';
    const title = document.createElement('b');
    title.textContent = it.title || 'Untitled';
    const sub = document.createElement('div');
    sub.className = 'hint';
    const when = it.ts ? new Date(it.ts * 1000).toLocaleString() : '';
    sub.textContent = [it.tab, when, (it.chars || 0) + ' chars'].filter(Boolean).join(' · ');
    const prev = document.createElement('div');
    prev.textContent = it.preview || '';
    left.append(title, sub, prev);
    left.onclick = () => openItem(it.id);
    const del = document.createElement('button');
    del.type = 'button'; del.textContent = 'Delete';
    del.onclick = async () => {
      if (!confirm('Delete this saved run?')) return;
      await fetch('/api/library', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({op:'delete', id: it.id})});
      renderLib();
    };
    li.append(left, del);
    ul.appendChild(li);
  });
}
async function openItem(id){
  const it = await (await fetch('/api/library?id=' + encodeURIComponent(id))).json();
  if (!it || it.error) return;
  $('#lib-modal').classList.remove('open');
  if (ST[it.tab] && ST[it.tab].out) { showTab(it.tab); paintOut(it.tab, it.text); toast('Opened in ' + it.tab); return; }
  if (it.tab === 'chat') { showTab('chat'); $('#ch-input').value = it.text; toast('Pasted into chat'); return; }
  if (it.tab === 'roleplay') { showTab('roleplay'); $('#rp-input').value = it.text; toast('Pasted into roleplay'); return; }
  showTab('caption');
  await copyText(it.text, ST.caption.status);
}
$('#open-lib').onclick = () => openLib();
$('#lib-close').onclick = () => $('#lib-modal').classList.remove('open');
$('#lib-q').oninput = () => { clearTimeout(libTimer); libTimer = setTimeout(renderLib, 200); };
$('#lib-filter').onchange = () => renderLib();
$('#open-keys').onclick = () => $('#keys-modal').classList.add('open');
$('#keys-close').onclick = () => $('#keys-modal').classList.remove('open');
['lib-modal','keys-modal'].forEach(id => { $('#' + id).onclick = (e) => { if (e.target.id === id) e.currentTarget.classList.remove('open'); }; });
function modalOpen(){ return $('#lib-modal').classList.contains('open') || $('#keys-modal').classList.contains('open'); }
document.addEventListener('keydown', (e) => {
  const meta = e.ctrlKey || e.metaKey;
  if (e.key === 'Escape') {
    if ($('#lib-modal').classList.contains('open')) { $('#lib-modal').classList.remove('open'); return; }
    if ($('#keys-modal').classList.contains('open')) { $('#keys-modal').classList.remove('open'); return; }
    stop(currentTab());
    return;
  }
  if (modalOpen()) return;
  if (meta && e.key === 'Enter') {
    const map = {ocr:'#ocr-run', translate:'#tr-run', prompt:'#pg-run', roleplay:'#rp-send', rewrite:'#rw-run', chat:'#ch-send', caption:'#cap-run', extract:'#ex-run', t2va:'#t2-next', ref2va:'#r2-run', sound:'#sd-run', orbit:'#or-run', character:'#cg-develop'};
    const btn = $(map[currentTab()]);
    if (btn) { e.preventDefault(); btn.click(); }
  } else if (e.altKey && /^[0-9]$/.test(e.key) && TABS[(+e.key + 9) % 10]) {
    e.preventDefault();
    showTab(TABS[(+e.key + 9) % 10]);
  } else if (meta && e.key.toLowerCase() === 'k') {
    e.preventDefault();
    if ($('#lib-modal').classList.contains('open')) $('#lib-modal').classList.remove('open'); else openLib(currentTab());
  } else if (meta && e.key === '/') {
    e.preventDefault();
    $('#keys-modal').classList.add('open');
  }
});
document.addEventListener('paste', async (e) => {
  const items = [...(e.clipboardData && e.clipboardData.items || [])].filter(i => i.type && i.type.startsWith('image/'));
  if (!items.length) return;
  const tab = currentTab();
  if (!TABS.includes(tab) || tab === 'rewrite') return;
  e.preventDefault();
  const files = items.map(i => i.getAsFile()).filter(Boolean);
  const imgs = [];
  for (const f of files) imgs.push(await shrinkImage(await readB64(f)));
  if (!imgs.length) return;
  if (tab === 'ocr') { OCR_IMGS.push(...imgs); drawOcr(); }
  else if (tab === 'translate') { TR_IMG = imgs[0]; drawTr(); if (imgs.length > 1) toast('Translate uses the first image'); }
  else if (tab === 'prompt') { PG_IMG = imgs[0]; drawPg(); if (imgs.length > 1) toast('The prompt uses the first image'); }
  else if (tab === 'roleplay') { RP_IMG = imgs[0]; drawRpImg(); }
  else if (tab === 'chat') { CHAT_IMG = imgs[0]; drawChatImg(); }
  else if (tab === 'caption') { imgs.forEach((b64, n) => CAPS.push({name: 'pasted-' + (CAPS.length + 1) + '.jpg', b64, text:'', status:''})); drawCaps(); }
  else if (tab === 'extract') { EX_IMG = imgs[0]; drawEx(); if (imgs.length > 1) toast('Extract uses the first image'); }
});

let OCR_IMGS = [];
const OCR_PRESETS = {
  'Extract all text': 'Extract all text. Keep the original line breaks.',
  'Markdown': 'Convert the document to markdown. Preserve headings, lists and reading order.',
  'Tables': 'Convert every table to a markdown table. Put other text after the tables.',
  'Handwriting': 'Transcribe the handwriting. Mark an uncertain word with [?]. Keep the wording that is actually on the page.',
  'Layout': 'Transcribe the page in reading order. Mark regions with markdown headings: header, body, footer, caption, stamp.'
};
function drawOcr(){ drawThumbs('ocr-thumbs', OCR_IMGS, i => { OCR_IMGS.splice(i, 1); drawOcr(); }); }
function drawOcrChips(){
  $('#ocr-presets').innerHTML = Object.keys(OCR_PRESETS).map(k => `<button type="button">${esc(k)}</button>`).join('');
  $$('#ocr-presets button').forEach(b => b.onclick = () => { $('#ocr-instr').value = OCR_PRESETS[b.textContent]; S('ocr').instr = $('#ocr-instr').value; persist(); });
}
setupDrop('ocr-drop', 'ocr-file', async files => {
  for (const f of files) {
    if (f.type === 'application/pdf' || /\.pdf$/i.test(f.name || '')) {
      if (!CFG.pdf) { ST.ocr.status.textContent = 'PDF needs pdftoppm (poppler-utils).'; continue; }
      ST.ocr.status.textContent = 'Converting ' + f.name + '…';
      const r = await (await fetch('/api/pdf', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({data: await readB64(f)})})).json();
      if (r.error) { ST.ocr.status.textContent = r.error; continue; }
      for (const p of r.pages || []) OCR_IMGS.push(await shrinkImage(p));
      ST.ocr.status.textContent = f.name + ': ' + (r.pages || []).length + ' page(s) added';
    } else if (/^image\//.test(f.type)) OCR_IMGS.push(await shrinkImage(await readB64(f)));
  }
  drawOcr();
});
$('#ocr-clear').onclick = () => { OCR_IMGS = []; drawOcr(); };
$('#ocr-instr').oninput = e => { S('ocr').instr = e.target.value; persist(); };
$('#ocr-lang').oninput = e => { S('ocr').lang = e.target.value; persist(); };
$('#ocr-run').onclick = () => busy($('#ocr-run'), async () => {
  if (!OCR_IMGS.length) { ST.ocr.status.textContent = 'Add an image or a PDF first.'; return; }
  const lang = $('#ocr-lang').value.trim();
  const instr = ($('#ocr-instr').value.trim() || 'Extract all text.') + (lang ? ' The text is in ' + lang + '.' : '');
  paintOut('ocr', '');
  let acc = '', completed = 0;
  for (let i = 0; i < OCR_IMGS.length; i++) {
    ST.ocr.label = OCR_IMGS.length > 1 ? ('Page ' + (i + 1) + '/' + OCR_IMGS.length) : '';
    const head = OCR_IMGS.length > 1 ? '\n\n<!-- page ' + (i + 1) + ' -->\n\n' : '';
    const prev = acc + head;
    const txt = await streamChat('ocr', withSys('ocr', [{role:'user', content: instr, images:[OCR_IMGS[i]]}]), t => paintOut('ocr', prev + t));
    acc = prev + txt;
    if (ST.ocr.last === 'ok') { completed++; continue; }
    break;
  }
  ST.ocr.label = '';
  const finalText = acc.trim();
  paintOut('ocr', finalText);
  if (completed && finalText) await remember('ocr', (finalText.split('\n').find(l => l.trim()) || 'OCR').slice(0, 80), finalText, {model: modelOf('ocr'), pages: completed});
});

function fillLangs(){
  $('#tr-src').innerHTML = ['Auto-detect', ...(CFG.languages || [])].map(l => `<option>${esc(l)}</option>`).join('');
  $('#tr-dst').innerHTML = (CFG.languages || []).map(l => `<option>${esc(l)}</option>`).join('');
  $('#tr-src').value = S('translate').src || 'Auto-detect';
  $('#tr-dst').value = S('translate').dst || 'German';
}
$('#tr-src').onchange = e => { S('translate').src = e.target.value; persist(); };
$('#tr-dst').onchange = e => { S('translate').dst = e.target.value; persist(); };
$('#tr-tone').onchange = e => { S('translate').tone = e.target.value; persist(); };
$('#tr-breaks').onchange = e => { S('translate').breaks = e.target.checked; persist(); };
$('#tr-gloss').oninput = e => { S('translate').glossary = e.target.value; persist(); };
$('#tr-swap').onclick = () => {
  const s = $('#tr-src').value, d = $('#tr-dst').value;
  if (![...$('#tr-src').options].some(o => o.value === d)) return;
  $('#tr-src').value = d;
  $('#tr-dst').value = s === 'Auto-detect' ? 'English' : s;
  if (ST.translate.out.value) { $('#tr-text').value = ST.translate.out.value; paintOut('translate', ''); }
  S('translate').src = $('#tr-src').value; S('translate').dst = $('#tr-dst').value; persist();
};
let TR_IMG = null;
function drawTr(){ drawThumbs('tr-thumbs', TR_IMG ? [TR_IMG] : [], () => { TR_IMG = null; drawTr(); }); }
setupDrop('tr-drop', 'tr-file', async files => { const imgs = await rememberPaste(files); if (imgs[0]) { TR_IMG = imgs[0].b64; drawTr(); } });
$('#tr-clear').onclick = () => { TR_IMG = null; drawTr(); };
function glossaryText(){
  const lines = ($('#tr-gloss').value || '').split('\n').map(s => s.trim()).filter(s => s.includes('='));
  if (!lines.length) return '';
  return '\nGlossary, use the right-hand side as written:\n' + lines.join('\n');
}
$('#tr-run').onclick = () => busy($('#tr-run'), async () => {
  const src = $('#tr-src').value, dst = $('#tr-dst').value, text = $('#tr-text').value.trim();
  const from = src === 'Auto-detect' ? '' : 'from ' + src + ' ';
  const tone = {formal:' Use a formal register.', casual:' Use a casual spoken register.'}[$('#tr-tone').value] || '';
  const breaks = $('#tr-breaks').checked ? ' Keep the original line breaks.' : '';
  const rules = ' Output only the translation. Keep names, and text inside quotes, unchanged except where the glossary maps them.' + tone + breaks + glossaryText();
  let msg;
  if (TR_IMG && !text) msg = {role:'user', content:'Translate the text in this image ' + from + 'into ' + dst + '.' + rules, images:[TR_IMG]};
  else if (text) msg = {role:'user', content:'Translate the following text ' + from + 'into ' + dst + '.' + rules + '\n\n' + text};
  else { ST.translate.status.textContent = 'Type some text or add an image.'; return; }
  const out = await runInto('translate', withSys('translate', [msg]));
  if (ST.translate.last !== 'err' && out.trim()) await remember('translate', (text || 'Image translation').slice(0, 80), out, {model: modelOf('translate'), to: dst});
});

const PG_PRESETS = {
  'Photoreal still': {lora:true, hint:'One dense photographic paragraph: subject, clothes, pose, place, light, lens.',
    instr:'Turn the idea into ONE photorealistic image prompt of 120 to 200 words. Cover subject, outfit, pose, expression, setting, lighting, camera (shot size, angle, lens, depth of field), color and texture. Natural prose, no lists, no headings.'},
  'Krea 2 still': {lora:true, hint:'One natural paragraph for Krea 2. With a character LoRA, describe pose, makeup, garments and the room.',
    instr:'Turn the idea into ONE Krea 2 prompt: a single natural-language paragraph, no tag list and no headings. Cover pose, expression, makeup, the garments that stay, the garments that come off, and the room, including the light.'},
  'LoRA dataset shot': {dataset:true, hint:'One training photo of the same person, plus a CAPTION line that starts with the trigger.',
    instr:'Write ONE prompt for a character-LoRA training image. Single photo, single person, no collage, no text in the image. Keep identity fixed and vary only what the idea asks. Specify framing, face angle, body orientation, camera height, outfit, makeup, background and lighting. The face stays sharp.'},
  'Qwen edit': {noAspect:true, hint:'One edit instruction: verb, the change, and what stays. Run the real edit in the Qwen Image 2.1 workflow.',
    instr:'Write ONE Qwen-Image-2.1 edit instruction as a single paragraph. Lead with an operation verb: Replace, Remove, Add, Change, Turn, Relight or Extend. Write as someone holding the input image. Say what the new thing is, where it sits, and its material, color and light. Close with one sentence that names what must stay, by role. Use affirmative sentences. Put readable text in double quotes. Output only the instruction.'},
  'Video (cinematic)': {hint:'Three or four shots with size, move, action and seconds.',
    instr:'Turn the idea into a cinematic video prompt: 3 or 4 shots. Each shot has a shot size, a camera move, a concrete visible action, and a duration in seconds.'},
  'MiniMax H3 draft': {hint:'A draft in the six-section H3 shape. Take production clips through H3 Prompter.',
    instr:'Write a MiniMax H3 draft using EXACTLY these sections, each as a lowercase header followed by a colon on its own line:\n\nsubject_definitions:\n<summary of each subject, mapped to <Picture N> when a reference is attached>\n\nsummary:\n<one paragraph overview>\n\nretention_analysis:\n<Subject 1>: fully_preserved.\n\ndetailed_description:\n[Shot 1] 0-4s ... [Shot 2] 4-8s ... Use concrete actions, a shot size and a camera move. Keep labels like <Subject 1> and <Picture 1> literal.\n\noverall_soundscape:\nSilent.\n\nnon_diegetic_music:\nNone.\n\nOutput only the draft.'}
};
const PG_LEGACY = {'Photoreal image':'Photoreal still', 'MiniMax H3 structured':'MiniMax H3 draft'};
let PG_PRESET = 'Photoreal still', PG_IMG = null;
function drawChips(id, presets, cur, onPick){
  $('#' + id).innerHTML = Object.keys(presets).map(k => `<button type="button" class="${k===cur?'active':''}">${esc(k)}</button>`).join('');
  $$('#' + id + ' button').forEach(b => b.onclick = () => onPick(b.textContent));
}
function pickPg(k){ PG_PRESET = k; S('prompt').preset = k; persist(); drawChips('pg-presets', PG_PRESETS, k, pickPg); $('#pg-preset-hint').textContent = PG_PRESETS[k].hint; }
function drawPg(){ drawThumbs('pg-thumbs', PG_IMG ? [PG_IMG] : [], () => { PG_IMG = null; drawPg(); }); }
setupDrop('pg-drop', 'pg-file', async files => { const imgs = await rememberPaste(files); if (imgs[0]) { PG_IMG = imgs[0].b64; drawPg(); } });
$('#pg-clear').onclick = () => { PG_IMG = null; drawPg(); };
$('#pg-count').oninput = e => { S('prompt').count = e.target.value; persist(); };
$('#pg-aspect').onchange = e => { S('prompt').aspect = e.target.value; persist(); };
$('#pg-trigger').oninput = e => { S('prompt').trigger = e.target.value; persist(); };
$('#pg-lora').onchange = e => { S('prompt').lora = e.target.checked; persist(); };
$('#pg-run').onclick = () => busy($('#pg-run'), async () => {
  const idea = $('#pg-idea').value.trim();
  if (!idea && !PG_IMG) { ST.prompt.status.textContent = 'Write an idea or add a reference image.'; return; }
  const preset = PG_PRESETS[PG_PRESET];
  const n = Math.max(1, Math.min(10, +$('#pg-count').value || 1));
  let content = preset.instr;
  if (PG_IMG && PG_PRESET === 'Qwen edit') content += '\n\nA reference image is attached. The instruction edits that image. Call it the image, and do not invent a written description of identity.';
  else if (PG_IMG) content += '\n\nA reference image is attached. Study it and keep the relevant details accurate.';
  if ($('#pg-lora').checked && preset.lora) content += '\n\nA character LoRA is loaded and already supplies face, hair, skin and body. Leave those out, even if an earlier sentence asked for them. Name the garments that stay and the garments that come off.';
  if (preset.dataset) content += '\n\nEnd with a final line that starts with CAPTION: then the exact trigger "' + ($('#pg-trigger').value.trim() || 'subject') + '" and only the elements that vary.';
  const aspect = $('#pg-aspect').value;
  if (aspect && !preset.noAspect) content += '\n\nFrame the image at ' + aspect + '.';
  const also = $('#pg-also').value.trim();
  if (also) content += '\n\nAlso include: ' + also;
  if (n > 1) content += '\n\nWrite ' + n + ' distinct variations, separated by a line containing only ---.';
  content += '\n\nIdea: ' + (idea || '(describe the reference and build on it)');
  const msg = {role:'user', content};
  if (PG_IMG) msg.images = [PG_IMG];
  const text = await runInto('prompt', withSys('prompt', [msg]));
  if (ST.prompt.last !== 'err' && text.trim()) await remember('prompt', idea || PG_PRESET, text, {model: modelOf('prompt'), preset: PG_PRESET});
});

// Modes, strengths and prompts come from perchance.org/wiz-text-rewriter (PC_RW).
// Studio extras are the older wiz-studio presets.
const RW_PRESETS = {
  'Screenplay': {instr:'Rewrite the text as a short screenplay beat. Action lines in the present tense. Dialogue under character names. Keep the original languages and the facts. Do not add a shot list.'},
  'Tighten dialogue': {instr:'Rewrite the dialogue so each line is shorter and more specific. Keep the speakers, the subtext and the facts. Leave action lines almost unchanged.'},
  'Subtext': {instr:'Rewrite so the feeling sits under the lines. Cut explanations of emotion. Keep the same events and the same languages.'},
  'German learner': {model:'sauerkraut-8B-Q8', hint:'Switches to sauerkraut-8B-Q8 when that model is installed.',
    instr:'I am learning German. Correct my German text. Reply in this format:\n\n**Korrigiert:** the corrected German text\n\n**Fehler:** a bullet list; for each mistake show the wrong part, then the correct part, and explain the rule briefly in English.\n\n**Tipp:** one short tip to remember.\n\nIf the text has no mistakes, say so and offer a more natural phrasing.'}
};
const RW_MODES = {};
String(PC_RW.modeOptions()).split('|||').filter(Boolean).forEach(p => { const i = p.indexOf('='); RW_MODES[p.slice(0, i)] = p.slice(i + 1); });
const RW_OLD = {'Formal':'formal', 'Casual':'casual', 'Shorter':'shorter', 'Longer':'longer', 'Fix grammar':'fix', 'Humanize':'humanize'};
let RW_PRESET = 'paraphrase';
function rwFill(){
  $('#rw-mode').innerHTML = '<optgroup label="Perchance modes">' + Object.entries(RW_MODES).map(([k, l]) => `<option value="${esc(k)}">${esc(l)}</option>`).join('') + '</optgroup>' +
    '<optgroup label="Studio extras">' + Object.keys(RW_PRESETS).map(k => `<option value="x:${esc(k)}">${esc(k)}</option>`).join('') + '</optgroup>';
  pcOpts('#rw-strength', PC_RW.strengthOptions());
}
function rwExtra(k){ return k.startsWith('x:') ? RW_PRESETS[k.slice(2)] : null; }
function pickRw(k){
  if (RW_OLD[k]) k = RW_OLD[k]; else if (RW_PRESETS[k]) k = 'x:' + k;
  if (!(k in RW_MODES) && !rwExtra(k)) k = 'paraphrase';
  RW_PRESET = k; S('rewrite').preset = k; persist();
  $('#rw-mode').value = k;
  const x = rwExtra(k);
  $('#rw-preset-hint').textContent = x ? (x.hint || 'Studio preset. Strength is ignored.') : '';
  if (x && x.model) setModel('rewrite', x.model);
}
function rwLabel(k){ const x = rwExtra(k); return x ? k.slice(2) : (RW_MODES[k] || k); }
$('#rw-mode').onchange = () => pickRw($('#rw-mode').value);
$('#rw-strength').onchange = () => { S('rewrite').strength = $('#rw-strength').value; persist(); };
$('#rw-extra').oninput = () => { S('rewrite').custom = $('#rw-extra').value; persist(); };
$('#rw-run').onclick = () => busy($('#rw-run'), async () => {
  const text = $('#rw-text').value.trim();
  if (!text) { ST.rewrite.status.textContent = 'Paste some text first.'; return; }
  const custom = $('#rw-extra').value.trim(), x = rwExtra(RW_PRESET);
  const content = x && !custom ? x.instr + '\n\nText:\n' + text : PC_RW.buildRewriter(text, x ? 'custom' : RW_PRESET, $('#rw-strength').value, custom);
  $('#rw-diffbox').hidden = true;
  const out = await runInto('rewrite', withSys('rewrite', [{role:'user', content}]));
  if (ST.rewrite.last !== 'err' && out.trim()) await remember('rewrite', rwLabel(RW_PRESET) + ': ' + text.slice(0, 60), out, {model: modelOf('rewrite'), preset: rwLabel(RW_PRESET), strength: $('#rw-strength').value});
});
$('#rw-clear').onclick = () => { $('#rw-text').value = ''; $('#rw-extra').value = ''; S('rewrite').custom = ''; persist(); $('#rw-diffbox').hidden = true; paintOut('rewrite', ''); };
$('#rw-diff').onclick = () => {
  const src = $('#rw-text').value, out = ST.rewrite.out.value;
  if (!src.trim() || !out.trim()) { ST.rewrite.status.textContent = 'Need both the source and a result.'; return; }
  const box = $('#rw-diffbox');
  box.hidden = false;
  box.innerHTML = wordDiffHtml(src, out);
};
$('#rw-use').onclick = () => { $('#rw-text').value = ST.rewrite.out.value; toast('Result is now the source'); };

let CHARS = [], CUR = null, HIST = [], RP_IMG = null;
function msgText(m){
  if (m && Array.isArray(m.alts) && m.alts.length) {
    const i = Math.min(m.alts.length - 1, Math.max(0, m.altIndex || 0));
    return m.alts[i] || '';
  }
  return (m && m.content) || '';
}
let STARTERS = [];
function charFields(){
  return {
    name: $('#rp-name').value.trim(),
    tags: $('#rp-tags').value.trim(),
    persona: $('#rp-persona').value,
    scenario: $('#rp-scenario').value,
    greeting: $('#rp-greeting').value,
    examples: $('#rp-examples').value,
    notes: $('#rp-notes').value
  };
}
function showChar(c){
  $('#rp-name').value = c && c.name || '';
  $('#rp-tags').value = c && c.tags || '';
  $('#rp-persona').value = c && c.persona || '';
  $('#rp-scenario').value = c && c.scenario || '';
  $('#rp-greeting').value = c && c.greeting || '';
  $('#rp-examples').value = c && c.examples || '';
  $('#rp-notes').value = c && c.notes || '';
  const title = $('#rp-chat-title');
  if (title) title.textContent = c && c.name ? ('· ' + c.name) : '';
}
function drawCharList(){
  const q = ($('#rp-filter').value || '').toLowerCase().trim();
  const list = $('#rp-list');
  list.innerHTML = '';
  const rows = CHARS.filter(c => {
    if (!q || c === CUR) return true;
    const hay = ((c.name || '') + ' ' + (c.tags || '') + ' ' + (c.notes || '')).toLowerCase();
    return hay.includes(q);
  });
  if (!rows.length) {
    const empty = document.createElement('div');
    empty.className = 'hint';
    empty.textContent = CHARS.length ? 'No characters match that filter.' : 'No characters yet. Click Add starters or New.';
    list.appendChild(empty);
    return;
  }
  rows.forEach(c => {
    const b = document.createElement('button');
    b.type = 'button';
    if (CUR && c.id === CUR.id) b.className = 'active';
    const name = document.createElement('span');
    name.className = 'cname';
    name.textContent = c.name || 'Untitled';
    b.appendChild(name);
    if (c.tags) {
      const tags = document.createElement('span');
      tags.className = 'ctags';
      tags.textContent = c.tags;
      b.appendChild(tags);
    }
    b.onclick = () => selectChar(c.id);
    list.appendChild(b);
  });
}
function saveChars(){ return fetch('/api/characters', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({characters: CHARS})}); }
function saveHist(){ if (CUR) fetch('/api/history', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: CUR.id, messages: HIST})}); }
function fillBasic(t){ const c = CUR || charFields(); return String(t || '').replaceAll('{{char}}', c.name || 'Character').replaceAll('{{user}}', $('#rp-user').value || 'User'); }
function fill(t){
  const c = CUR || charFields();
  const ex = (c.examples || '').trim();
  const exBlock = ex ? ('\n\nExample dialogue:\n' + ex) : '';
  return fillBasic(String(t || '').replaceAll('{{persona}}', c.persona || '').replaceAll('{{scenario}}', c.scenario || '').replaceAll('{{examples}}', exBlock));
}
function roleSys(){
  let sys = fill(sysOf('roleplay'));
  const ex = ((CUR && CUR.examples) || '').trim();
  if (ex && !sys.includes(ex)) sys += '\n\nExample dialogue:\n' + fillBasic(ex);
  return sys;
}
function toApi(m){ const o = {role:m.role, content: msgText(m)}; if (m.images && m.images[0]) o.images = [m.images[0]]; return o; }
function drawRpImg(){ drawThumbs('rp-thumbs', RP_IMG ? [RP_IMG] : [], () => { RP_IMG = null; drawRpImg(); }); }
function fillRpBody(el, text){
  el.textContent = '';
  String(text || '').split('\n').forEach((line, li) => {
    if (li) el.appendChild(document.createElement('br'));
    String(line).split(/(\*[^*]+\*)/g).forEach(part => {
      if (part.length > 2 && part.startsWith('*') && part.endsWith('*')) {
        const em = document.createElement('em');
        em.textContent = part.slice(1, -1);
        el.appendChild(em);
      } else if (part) {
        el.appendChild(document.createTextNode(part));
      }
    });
  });
}
function drawLog(extra){
  const log = $('#rp-log');
  const name = (CUR || charFields()).name || 'Character';
  const you = $('#rp-user').value || 'You';
  const msgs = extra || HIST;
  const streaming = !!extra;
  log.innerHTML = '';
  if (!msgs.length) {
    const empty = document.createElement('div');
    empty.className = 'msg rp-empty';
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = CUR ? name : 'Roleplay';
    empty.appendChild(who);
    const body = document.createElement('div');
    body.className = 'body';
    if (!CUR) {
      body.textContent = 'Pick a character on the left, or create one. Then send a message.';
    } else {
      const bits = [];
      if (CUR.tags) bits.push(CUR.tags);
      if (CUR.scenario) bits.push(fillBasic(CUR.scenario).slice(0, 220));
      body.textContent = (bits.join('\n\n') || 'Save the card, then send a message.') + '\n\nTip: Enter sends · Shift+Enter makes a new line.';
    }
    empty.appendChild(body);
    log.appendChild(empty);
    return;
  }
  msgs.forEach((m, i) => {
    const text = msgText(m);
    const isContinue = m.role === 'user' && (m.continue || text === '*continues*' || text === '(continue)');
    const div = document.createElement('div');
    div.className = 'msg ' + (m.role === 'user' ? 'user' : 'assistant') + (isContinue ? ' rp-continue' : '');
    div.dataset.i = String(i);
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = isContinue ? 'Continue' : (m.role === 'user' ? you : name);
    div.appendChild(who);
    if (m.images && m.images[0]) {
      const img = document.createElement('img');
      img.alt = '';
      img.src = 'data:' + sniffMime(m.images[0]) + ';base64,' + m.images[0];
      img.style.cssText = 'height:72px;border-radius:6px;display:block;margin-bottom:4px';
      div.appendChild(img);
    }
    const body = document.createElement('div');
    body.className = 'body';
    if (isContinue) body.textContent = 'Keep going…';
    else fillRpBody(body, text);
    div.appendChild(body);
    if (!streaming) {
      const tools = document.createElement('div');
      tools.className = 'tools';
      const copy = document.createElement('button');
      copy.type = 'button'; copy.textContent = 'Copy'; copy.dataset.act = 'copy';
      const edit = document.createElement('button');
      edit.type = 'button'; edit.textContent = 'Edit'; edit.dataset.act = 'edit';
      const del = document.createElement('button');
      del.type = 'button'; del.textContent = 'Delete'; del.dataset.act = 'del';
      if (isContinue) tools.append(del);
      else tools.append(copy, edit, del);
      if (i === HIST.length - 1 && m.role === 'assistant' && (m.alts || []).length > 1) {
        const prev = document.createElement('button');
        prev.type = 'button'; prev.textContent = '‹'; prev.dataset.swipe = '-1';
        const label = document.createElement('span');
        label.className = 'swipe';
        label.textContent = ((m.altIndex || 0) + 1) + '/' + m.alts.length;
        const next = document.createElement('button');
        next.type = 'button'; next.textContent = '›'; next.dataset.swipe = '1';
        tools.append(prev, label, next);
      }
      div.appendChild(tools);
    }
    log.appendChild(div);
  });
  log.scrollTop = log.scrollHeight;
}
function startEdit(i){
  const m = HIST[i];
  const div = $('#rp-log').querySelector('.msg[data-i="' + i + '"]');
  if (!m || !div) return;
  const ta = document.createElement('textarea');
  ta.style.width = '100%';
  ta.value = msgText(m);
  const bar = document.createElement('div');
  bar.className = 'bar';
  const save = document.createElement('button');
  save.type = 'button'; save.className = 'primary'; save.textContent = 'Save edit';
  const cancel = document.createElement('button');
  cancel.type = 'button'; cancel.textContent = 'Cancel';
  save.onclick = (ev) => {
    ev.stopPropagation();
    if (i < HIST.length - 1 && !confirm('This drops the messages after the one you edited.')) return;
    const text = ta.value;
    if (m.alts && m.alts.length) m.alts[m.altIndex || 0] = text;
    m.content = text;
    HIST = HIST.slice(0, i + 1);
    saveHist();
    drawLog();
  };
  cancel.onclick = (ev) => { ev.stopPropagation(); drawLog(); };
  const body = div.querySelector('.body');
  if (body) body.replaceWith(ta); else div.appendChild(ta);
  bar.append(save, cancel);
  ta.after(bar);
  ta.focus();
}
function swipe(dir){
  const m = HIST[HIST.length - 1];
  if (!m || m.role !== 'assistant' || !(m.alts || []).length) return;
  let i = (m.altIndex || 0) + dir;
  if (i < 0) i = m.alts.length - 1;
  if (i >= m.alts.length) i = 0;
  m.altIndex = i;
  m.content = m.alts[i];
  saveHist();
  drawLog();
}
function deleteMsg(i){
  if (i < 0 || i >= HIST.length) return;
  if (i < HIST.length - 1 && !confirm('Delete this message and everything after it?')) return;
  HIST = HIST.slice(0, i);
  saveHist();
  drawLog();
}
$('#rp-log').onclick = (e) => {
  const b = e.target.closest('button');
  if (!b || !$('#rp-log').contains(b) || b.dataset.act == null && b.dataset.swipe == null) return;
  const host = b.closest('.msg');
  if (!host) return;
  const i = +host.dataset.i;
  if (b.dataset.act === 'copy') copyText(msgText(HIST[i]), ST.roleplay.status);
  else if (b.dataset.act === 'edit') startEdit(i);
  else if (b.dataset.act === 'del') deleteMsg(i);
  else if (b.dataset.swipe) swipe(+b.dataset.swipe);
};
function mergeStarters(list){
  const have = new Set(CHARS.map(c => c.id));
  let added = 0;
  (list || STARTERS || []).forEach(s => {
    if (!s || !s.id || have.has(s.id)) return;
    CHARS.push(Object.assign({}, s));
    have.add(s.id);
    added += 1;
  });
  return added;
}
async function loadChars(){
  try {
    const data = await (await fetch('/api/characters')).json();
    CHARS = data.characters || [];
    STARTERS = data.starters || [];
  } catch (e) { CHARS = []; STARTERS = []; }
  if (!CHARS.length) {
    mergeStarters(STARTERS);
    if (CHARS.length) await saveChars();
  }
  drawCharList();
  await selectChar(S('roleplay').char || (CHARS[0] && CHARS[0].id));
}
async function selectChar(id){
  CUR = CHARS.find(c => c.id === id) || CHARS[0] || null;
  showChar(CUR);
  drawCharList();
  $('#rp-input').value = '';
  RP_IMG = null;
  if (typeof drawRpImg === 'function') drawRpImg();
  if (!CUR) { HIST = []; drawLog(); return; }
  S('roleplay').char = CUR.id; persist();
  try { HIST = (await (await fetch('/api/history?id=' + encodeURIComponent(CUR.id))).json()).messages || []; }
  catch (e) { HIST = []; }
  if (!HIST.length && CUR.greeting) HIST = [{role:'assistant', content: fill(CUR.greeting), alts:[fill(CUR.greeting)], altIndex:0}];
  drawLog();
  const card = $('#rp-card');
  if (card && CUR && CUR.persona) card.open = false;
}
async function completeAssistant(base){
  const live = {role:'assistant', content:''};
  const txt = await streamChat('roleplay', withSys('roleplay', base.map(toApi), roleSys()), t => { live.content = t; drawLog(base.concat([live])); });
  return txt.trim();
}
async function reply(){
  if (!CUR) { ST.roleplay.status.textContent = 'Save the character card first.'; return; }
  const txt = await completeAssistant(HIST);
  if (txt) HIST.push({role:'assistant', content: txt, alts:[txt], altIndex:0});
  saveHist();
  drawLog();
}
$('#rp-filter').oninput = () => drawCharList();
$('#rp-user').oninput = e => { S('roleplay').user = e.target.value; persist(); drawLog(); };
$('#rp-new').onclick = () => {
  CUR = null;
  showChar(null);
  HIST = [];
  drawCharList();
  drawLog();
  const card = $('#rp-card');
  if (card) card.open = true;
  $('#rp-name').focus();
};
$('#rp-dup').onclick = async () => {
  if (!CUR) { ST.roleplay.status.textContent = 'Pick a character to duplicate.'; return; }
  const copy = Object.assign({}, CUR, {
    id: (CUR.name || 'char').toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') + '-' + Date.now().toString(36),
    name: (CUR.name || 'Character') + ' copy'
  });
  CHARS.push(copy);
  await saveChars();
  await selectChar(copy.id);
  const card = $('#rp-card');
  if (card) card.open = true;
  ST.roleplay.status.textContent = 'Duplicated. Rename and save if you like.';
};
$('#rp-starters').onclick = async () => {
  const n = mergeStarters(STARTERS);
  if (!n) { ST.roleplay.status.textContent = 'All starter characters are already in your list.'; return; }
  await saveChars();
  drawCharList();
  ST.roleplay.status.textContent = 'Added ' + n + ' starter character' + (n === 1 ? '' : 's') + '.';
  if (!CUR && CHARS[0]) await selectChar(CHARS[0].id);
};
$('#rp-save').onclick = async () => {
  const f = charFields();
  if (!f.name) { ST.roleplay.status.textContent = 'Give the character a name.'; return; }
  if (CUR) Object.assign(CUR, f);
  else {
    CUR = Object.assign({id: f.name.toLowerCase().replace(/[^a-z0-9]+/g, '-').replace(/^-|-$/g, '') + '-' + Date.now().toString(36)}, f);
    CHARS.push(CUR);
    if (f.greeting && !HIST.length) HIST = [{role:'assistant', content: fill(f.greeting), alts:[fill(f.greeting)], altIndex:0}];
  }
  await saveChars();
  drawCharList();
  drawLog();
  saveHist();
  S('roleplay').char = CUR.id;
  persist();
  ST.roleplay.status.textContent = 'Card saved.';
};
$('#rp-del').onclick = async () => {
  if (!CUR || !confirm('Delete ' + CUR.name + ' and their chat history?')) return;
  const gone = CUR;
  HIST = []; saveHist();
  CHARS = CHARS.filter(c => c !== gone);
  CUR = null;
  await saveChars();
  if (CHARS[0]) await selectChar(CHARS[0].id); else { drawCharList(); showChar(null); drawLog(); }
};
$('#rp-stop').onclick = () => stop('roleplay');
$('#rp-send').onclick = () => busy($('#rp-send'), async () => {
  const t = $('#rp-input').value.trim();
  if (!t && !RP_IMG) return;
  if (!CUR) { ST.roleplay.status.textContent = 'Save the character card first.'; return; }
  const msg = {role:'user', content: t};
  if (RP_IMG) msg.images = [RP_IMG];
  HIST.push(msg);
  $('#rp-input').value = '';
  RP_IMG = null;
  drawRpImg();
  drawLog();
  saveHist();
  await reply();
});
$('#rp-continue').onclick = () => busy($('#rp-continue'), async () => {
  if (!CUR) { ST.roleplay.status.textContent = 'Save the character card first.'; return; }
  if (!HIST.length) { ST.roleplay.status.textContent = 'Send a message first.'; return; }
  HIST.push({role:'user', content:'*continues*', continue:true});
  drawLog();
  saveHist();
  await reply();
});
$('#rp-input').onkeydown = e => {
  if (e.key !== 'Enter') return;
  if (e.shiftKey) return;
  e.preventDefault();
  $('#rp-send').click();
};
$('#rp-regen').onclick = () => busy($('#rp-regen'), async () => {
  if (!CUR) { ST.roleplay.status.textContent = 'Save the character card first.'; return; }
  if (!HIST.length) return;
  const base = HIST.slice();
  let host = null;
  if (base[base.length - 1].role === 'assistant') host = base.pop();
  const txt = await completeAssistant(base);
  if (!txt || ST.roleplay.last === 'err') { drawLog(); return; }
  if (host) {
    if (!host.alts || !host.alts.length) host.alts = [msgText(host) || host.content || ''];
    host.alts.push(txt);
    host.alts = host.alts.slice(-8);
    host.altIndex = host.alts.length - 1;
    host.content = txt;
    HIST = base.concat([host]);
  } else HIST = base.concat([{role:'assistant', content: txt, alts:[txt], altIndex:0}]);
  saveHist();
  drawLog();
});
$('#rp-edit').onclick = () => {
  const i = HIST.map(m => m.role).lastIndexOf('user');
  if (i < 0) return;
  $('#rp-input').value = msgText(HIST[i]);
  RP_IMG = (HIST[i].images && HIST[i].images[0]) || null;
  drawRpImg();
  HIST = HIST.slice(0, i);
  drawLog();
  saveHist();
  $('#rp-input').focus();
};
$('#rp-reset').onclick = () => {
  if (!confirm('Clear this chat and start again from the greeting?')) return;
  const greet = CUR && CUR.greeting ? fill(CUR.greeting) : '';
  HIST = greet ? [{role:'assistant', content: greet, alts:[greet], altIndex:0}] : [];
  saveHist();
  drawLog();
};
$('#rp-export').onclick = () => {
  const name = (CUR && CUR.name) || 'chat';
  const you = $('#rp-user').value || 'You';
  download(name + '-' + stamp() + '.md', HIST.map(m => '**' + (m.role === 'user' ? you : name) + ':** ' + msgText(m)).join('\n\n'));
};
$('#rp-img').onclick = () => $('#rp-file').click();
$('#rp-file').onchange = async () => {
  const f = $('#rp-file').files && $('#rp-file').files[0];
  $('#rp-file').value = '';
  if (!f) return;
  RP_IMG = await shrinkImage(await readB64(f));
  drawRpImg();
};
function cardFromJson(j){
  const data = j && (j.data || j);
  if (!data || !data.name) throw new Error('The card needs a name.');
  const persona = data.persona || [data.description, data.personality].filter(Boolean).join('\n\n');
  if (!persona && !data.scenario && !data.first_mes && !data.greeting) throw new Error('The card needs a persona or a scenario.');
  return {
    name: data.name || data.char_name,
    tags: data.tags || '',
    persona: persona || '',
    scenario: data.scenario || '',
    greeting: data.greeting || data.first_mes || '',
    examples: data.examples || data.mes_example || '',
    notes: data.notes || ''
  };
}
$('#rp-export-card').onclick = () => {
  const f = CUR ? Object.assign({}, CUR) : charFields();
  if (!f.name) { ST.roleplay.status.textContent = 'Give the character a name.'; return; }
  download(f.name.replace(/[^\w.-]+/g, '_') + '.json', JSON.stringify({name:f.name, tags:f.tags||'', persona:f.persona||'', scenario:f.scenario||'', greeting:f.greeting||'', examples:f.examples||'', notes:f.notes||''}, null, 2));
};
$('#rp-import-card').onclick = () => $('#rp-import-file').click();
$('#rp-import-file').onchange = async () => {
  const f = $('#rp-import-file').files && $('#rp-import-file').files[0];
  $('#rp-import-file').value = '';
  if (!f) return;
  try {
    const card = cardFromJson(JSON.parse(await f.text()));
    CUR = null;
    HIST = [];
    showChar(card);
    const details = $('#rp-card');
    if (details) details.open = true;
    $('#rp-save').click();
  } catch (e) { ST.roleplay.status.textContent = e.message; }
};

let THREADS = [], THREAD = null, CHAT = [], CHAT_IMG = null;
function drawChatImg(){ drawThumbs('ch-thumbs', CHAT_IMG ? [CHAT_IMG] : [], () => { CHAT_IMG = null; drawChatImg(); }); }
function drawThreads(){
  const q = ($('#ch-filter').value || '').toLowerCase();
  const list = $('#ch-list');
  list.innerHTML = '';
  THREADS.filter(t => !q || (t.title || '').toLowerCase().includes(q) || t === THREAD).forEach(t => {
    const b = document.createElement('button');
    b.type = 'button';
    b.textContent = t.title || 'New chat';
    if (THREAD && t.id === THREAD.id) b.className = 'active';
    b.onclick = () => selectThread(t.id);
    list.appendChild(b);
  });
}
function saveThreads(){ return fetch('/api/threads', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({threads: THREADS})}); }
function saveChat(){
  if (!THREAD) return;
  THREAD.title = $('#ch-title').value.trim() || 'New chat';
  THREAD.system = $('#ch-sys').value;
  saveThreads();
  return fetch('/api/thread', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: THREAD.id, messages: CHAT})});
}
function drawChat(extra){
  const log = $('#ch-log');
  const msgs = extra || CHAT;
  log.innerHTML = '';
  if (!msgs.length) {
    const p = document.createElement('div');
    p.className = 'hint';
    p.textContent = 'A plain thread, separate from roleplay. Paste an image when the model can see.';
    log.appendChild(p);
    return;
  }
  msgs.forEach(m => {
    const div = document.createElement('div');
    div.className = 'msg ' + m.role;
    const who = document.createElement('span');
    who.className = 'who';
    who.textContent = m.role === 'user' ? 'You' : 'Assistant';
    div.appendChild(who);
    if (m.images && m.images[0]) {
      const img = document.createElement('img');
      img.alt = '';
      img.src = 'data:' + sniffMime(m.images[0]) + ';base64,' + m.images[0];
      img.style.cssText = 'height:72px;border-radius:6px;display:block;margin-bottom:4px';
      div.appendChild(img);
    }
    const body = document.createElement('div');
    body.textContent = m.content || '';
    div.appendChild(body);
    log.appendChild(div);
  });
  log.scrollTop = log.scrollHeight;
}
async function loadThreads(){
  try { THREADS = (await (await fetch('/api/threads')).json()).threads || []; } catch (e) { THREADS = []; }
  if (!THREADS.length) {
    THREADS = [{id:'chat-' + Date.now().toString(36), title:'New chat', system:''}];
    await saveThreads();
  }
  await selectThread(S('chat').thread || THREADS[0].id);
}
async function selectThread(id){
  THREAD = THREADS.find(t => t.id === id) || THREADS[0] || null;
  if (!THREAD) { CHAT = []; drawThreads(); drawChat(); return; }
  S('chat').thread = THREAD.id; persist();
  $('#ch-title').value = THREAD.title || '';
  $('#ch-sys').value = THREAD.system || '';
  $('#ch-input').value = '';
  CHAT_IMG = null;
  if (typeof drawChatImg === 'function') drawChatImg();
  try { CHAT = (await (await fetch('/api/thread?id=' + encodeURIComponent(THREAD.id))).json()).messages || []; }
  catch (e) { CHAT = []; }
  drawThreads();
  drawChat();
}
async function chatReply(){
  const base = CHAT.slice();
  const live = {role:'assistant', content:''};
  const sys = ($('#ch-sys').value || '').trim() || sysOf('chat');
  const txt = await streamChat('chat', withSys('chat', base.map(m => { const o = {role:m.role, content:m.content || ''}; if (m.images && m.images[0]) o.images = m.images; return o; }), sys), t => { live.content = t; drawChat(base.concat([live])); });
  if (txt.trim()) CHAT.push({role:'assistant', content: txt.trim()});
  saveChat();
  drawChat();
}
$('#ch-filter').oninput = () => drawThreads();
$('#ch-title').oninput = () => { if (!THREAD) return; THREAD.title = $('#ch-title').value.trim() || 'New chat'; drawThreads(); saveThreads(); };
$('#ch-sys').oninput = () => { if (THREAD) { THREAD.system = $('#ch-sys').value; saveThreads(); } };
$('#ch-new').onclick = async () => {
  const t = {id:'chat-' + Date.now().toString(36), title:'New chat', system:''};
  THREADS.unshift(t);
  await saveThreads();
  CHAT = [];
  await fetch('/api/thread', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id: t.id, messages: []})});
  await selectThread(t.id);
  $('#ch-title').focus();
};
$('#ch-del').onclick = async () => {
  if (!THREAD || !confirm('Delete this thread?')) return;
  const id = THREAD.id;
  THREADS = THREADS.filter(t => t.id !== id);
  THREAD = null;
  await saveThreads();
  await fetch('/api/thread', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({id, messages: []})});
  if (!THREADS.length) await $('#ch-new').onclick();
  else await selectThread(THREADS[0].id);
};
$('#ch-stop').onclick = () => stop('chat');
$('#ch-send').onclick = () => busy($('#ch-send'), async () => {
  const t = $('#ch-input').value.trim();
  if (!t && !CHAT_IMG) return;
  if (!THREAD) await $('#ch-new').onclick();
  const msg = {role:'user', content: t};
  if (CHAT_IMG) msg.images = [CHAT_IMG];
  CHAT.push(msg);
  $('#ch-input').value = '';
  CHAT_IMG = null;
  drawChatImg();
  drawChat();
  saveChat();
  await chatReply();
});
$('#ch-input').onkeydown = e => { if (e.key === 'Enter' && (e.ctrlKey || e.metaKey)) { e.preventDefault(); $('#ch-send').click(); } };
$('#ch-regen').onclick = () => busy($('#ch-regen'), async () => {
  if (CHAT.length && CHAT[CHAT.length - 1].role === 'assistant') CHAT.pop();
  if (!(CHAT.length && CHAT[CHAT.length - 1].role === 'user')) { drawChat(); return; }
  await chatReply();
});
$('#ch-edit').onclick = () => {
  const i = CHAT.map(m => m.role).lastIndexOf('user');
  if (i < 0) return;
  $('#ch-input').value = CHAT[i].content || '';
  CHAT_IMG = (CHAT[i].images && CHAT[i].images[0]) || null;
  drawChatImg();
  CHAT = CHAT.slice(0, i);
  saveChat();
  drawChat();
  $('#ch-input').focus();
};
$('#ch-clearimg').onclick = () => { CHAT_IMG = null; drawChatImg(); };
$('#ch-export').onclick = () => {
  const title = (THREAD && THREAD.title) || 'chat';
  download(title.replace(/[^\w.-]+/g, '_') + '-' + stamp() + '.md', CHAT.map(m => '**' + m.role + ':** ' + (m.content || '')).join('\n\n'));
};
$('#ch-lib').onclick = async () => {
  const text = CHAT.map(m => (m.role === 'user' ? 'You' : 'Assistant') + ': ' + (m.content || '')).join('\n\n');
  if (!text.trim()) { ST.chat.status.textContent = 'The thread is empty.'; return; }
  await remember('chat', (THREAD && THREAD.title) || 'Chat', text, {model: modelOf('chat')});
  toast('Saved to the library');
};

let CAPS = [], CAP_BUSY = false, CAP_PRESET = 'Training';
const CAP_PRESETS = {
  'Training': {hint:'Trigger first, then framing, pose, clothes, makeup, background and light. Face, hair, skin and body stay out of the caption.',
    build(trigger){ return 'Write one short training caption for this photo. Start with the exact trigger "' + (trigger || 'subject') + '". Describe framing, pose, expression, clothing, makeup, background and lighting. Leave face structure, hair, skin tone and body out of the caption. Output only the caption.'; }},
  'Natural': {hint:'One or two concrete sentences about who, clothing, pose, place and light.',
    build(){ return 'Describe this image in one or two concrete sentences: who is present, clothing, pose, place and light. Output only the description.'; }}
};
function pickCap(k){ CAP_PRESET = k; S('caption').preset = k; persist(); drawChips('cap-presets', CAP_PRESETS, k, pickCap); $('#cap-hint').textContent = CAP_PRESETS[k].hint; }
function sidecarName(name){
  const base = String(name || 'image').replace(/\.[^.]+$/, '').replace(/[^\w.\- ]+/g, '_').replace(/^_+|_+$/g, '').slice(0, 80) || 'image';
  return base + '.txt';
}
function drawCaps(){
  const root = $('#cap-list');
  root.innerHTML = '';
  if (!CAPS.length) {
    const p = document.createElement('div');
    p.className = 'hint';
    p.textContent = 'Drop a batch of images. Each caption becomes a sidecar .txt in the zip.';
    root.appendChild(p);
    return;
  }
  CAPS.forEach((c, i) => {
    const row = document.createElement('div');
    row.className = 'cap-row';
    const img = document.createElement('img');
    img.alt = c.name || '';
    img.src = 'data:' + sniffMime(c.b64) + ';base64,' + c.b64;
    const col = document.createElement('div');
    const name = document.createElement('div');
    name.className = 'hint';
    name.textContent = (c.name || 'image') + (c.status ? ' · ' + c.status : '');
    const ta = document.createElement('textarea');
    ta.value = c.text || '';
    ta.oninput = () => { c.text = ta.value; };
    const rm = document.createElement('button');
    rm.type = 'button'; rm.textContent = 'Remove';
    rm.onclick = () => { if (CAP_BUSY) return; CAPS.splice(i, 1); drawCaps(); };
    col.append(name, ta, rm);
    row.append(img, col);
    root.appendChild(row);
  });
}
function setCapRow(i, status, text){
  const c = CAPS[i];
  if (!c) return;
  c.status = status;
  if (text != null) c.text = text;
  const row = $('#cap-list').children[i];
  if (!row) return;
  const hint = row.querySelector('.hint');
  if (hint) hint.textContent = (c.name || 'image') + ' · ' + status;
  const ta = row.querySelector('textarea');
  if (ta && document.activeElement !== ta) ta.value = c.text || '';
  row.scrollIntoView({block:'nearest'});
}
setupDrop('cap-drop', 'cap-file', async files => {
  const imgs = await rememberPaste(files);
  imgs.forEach(img => CAPS.push({name: img.name, b64: img.b64, text:'', status:''}));
  drawCaps();
  ST.caption.status.textContent = CAPS.length + ' image(s) ready';
});
$('#cap-trigger').oninput = e => { S('caption').trigger = e.target.value; persist(); };
$('#cap-clear').onclick = () => { if (CAP_BUSY) return; CAPS = []; drawCaps(); };
$('#cap-stop').onclick = () => { stop('caption'); };
$('#cap-run').onclick = () => busy($('#cap-run'), async () => {
  if (!CAPS.length) { ST.caption.status.textContent = 'Add images first.'; return; }
  CAP_BUSY = true;
  drawCaps();
  const trigger = $('#cap-trigger').value.trim();
  const instr = CAP_PRESETS[CAP_PRESET].build(trigger);
  let done = 0;
  try {
    for (let i = 0; i < CAPS.length; i++) {
      ST.caption.label = (i + 1) + '/' + CAPS.length + ' · ' + CAPS[i].name;
      setCapRow(i, 'working');
      const txt = await streamChat('caption', withSys('caption', [{role:'user', content: instr, images:[CAPS[i].b64]}]), t => setCapRow(i, 'streaming', t));
      if (ST.caption.last === 'abort') { setCapRow(i, 'stopped', txt); break; }
      if (ST.caption.last === 'err') { setCapRow(i, 'failed', txt); break; }
      setCapRow(i, 'done', txt.trim());
      done++;
    }
  } finally {
    ST.caption.label = '';
    CAP_BUSY = false;
  }
  ST.caption.status.textContent = done + '/' + CAPS.length + ' captioned';
  const joined = CAPS.filter(c => (c.text || '').trim()).map(c => sidecarName(c.name).replace(/\.txt$/, '') + '\n' + c.text.trim()).join('\n\n');
  if (joined) await remember('caption', 'Captions · ' + done, joined, {model: modelOf('caption'), preset: CAP_PRESET, count: done});
});
$('#cap-zip').onclick = async () => {
  const files = CAPS.filter(c => (c.text || '').trim()).map(c => ({name: sidecarName(c.name), text: c.text.trim()}));
  if (!files.length) { ST.caption.status.textContent = 'No captions yet.'; return; }
  const r = await (await fetch('/api/zip', {method:'POST', headers:{'Content-Type':'application/json'}, body: JSON.stringify({files})})).json();
  if (r.error) { ST.caption.status.textContent = r.error; return; }
  downloadB64('captions-' + stamp() + '.zip', r.zip, 'application/zip');
  toast('Zip downloaded');
};

let EX_IMG = null, EX_PRESET = 'Camera & blocking';
const EX_PRESETS = {
  'Camera & blocking': {
    hint: 'Camera height, angle and lens, where each person stands, and one H3 framing line.',
    instr: 'Study the attached image and extract camera and character position only.\n\n' +
      'Use these words when they fit. Do not invent a closer crop than the frame shows.\n' +
      '- Shot size: extreme close-up, close-up, medium close-up (chest up), medium shot (waist up), medium-wide (knees up), wide shot, extreme wide shot.\n' +
      '- Camera height: eye level, slightly low, slightly high, overhead.\n' +
      '- Facing: front-facing toward the camera, three-quarter view, clean side profile, from behind, over the shoulder.\n' +
      '- Lens feel: 85mm compressed for a close view, 50mm for a medium view. Call the frame wide only when it is genuinely wide and faces are small. Never write wide-angle, fisheye, 24mm or GoPro on a shot where a face is readable.\n\n' +
      'Write these markdown sections:\n\n' +
      '## Camera\n' +
      '- Shot size\n- Height\n- Angle and facing\n- Lens feel\n- Distance from the subject (close, a few feet, across the room)\n\n' +
      '## Character position\n' +
      'If nobody is in the frame, say "no person in frame" and place the main object instead.\n' +
      'Otherwise, for each person from left to right:\n' +
      '- Where they sit in the frame (left third, center, right third, or near an edge) and how much of the body is in frame\n' +
      '- Depth (foreground, midground, background) and what, if anything, is in front of them\n' +
      '- Body orientation (square to camera, turned left or right, back to camera) and roughly how far they are turned\n' +
      '- Head and gaze (toward the lens, off to one side, down, toward another person)\n' +
      '- Headroom and the space on the look side\n' +
      '- Relation to furniture, a door, or another person (standing at, seated on, beside, behind)\n\n' +
      '## Depth\n- Foreground\n- Background\n\n' +
      '## H3 line\n' +
      'One sentence in this shape, filled with what the frame actually shows: "framed at <height> in a <shot size>, <facing>, shot on a <lens> with <perspective>, <placement in the frame>."\n' +
      'Do not add a camera move. This is a still.'
  },
  'Outfit': {
    hint: 'Each garment, fabric, color and how it is worn, then one pasteable outfit paragraph.',
    instr: 'Extract the clothing only.\n\n' +
      'For each visible garment, cloth accessory (scarf, belt, dupatta, jacket) and the footwear, write a bullet with: the name of the piece, color and pattern, fabric if the photo shows it, and the cut (neckline, sleeves, length, fit, drape, layers, and whether it is tucked, open, tied or slipping). Say what is cropped.\n' +
      'Skip face, hair, skin, jewellery and the room, except one clause when a garment touches them (a saree pallu over the shoulder).\n' +
      'After the bullets, write one paragraph that starts with "Outfit:" and can be pasted into an image prompt. Name each garment once.'
  },
  'Jewellery': {
    hint: 'Every visible piece, its material, and where it sits. Says none when the frame has none.',
    instr: 'Extract jewellery and worn metal or stone ornaments only. Include earrings, necklaces, nose pins, bangles, bracelets, rings, waist chains, maang tikkas, hair jewellery, anklets, brooches and watches.\n\n' +
      'For each piece write a bullet: type, metal or material, stones or color, size relative to the feature it sits on, and exact placement (which ear, which wrist, which finger, how it rests).\n' +
      'If a side is cropped, say so. If nothing is visible, write "Jewellery: none visible." and stop.\n' +
      'Do not describe clothes, makeup or the face. End with one line that starts with "Jewellery:" and lists the pieces in wearing order, each named once.'
  },
  'Makeup': {
    hint: 'Skin finish, brows, eyes, cheeks and lips. Placement only, no product brands.',
    instr: 'Extract the visible makeup. Do not name brands.\n\n' +
      'Cover, and write "not visible" when the crop hides it: skin finish (matte, satin, dewy, bare); coverage if it is obvious; blush color and where it sits; highlight or contour only if you can see it; brow shape and fill; eye shadow, liner and lashes; lip color and finish; bindi or other face paint that is makeup rather than jewellery.\n' +
      'End with one paragraph that starts with "Makeup:" and can be pasted into a close-up prompt. Do not describe identity, hair, outfit or the room.'
  },
  'Hair': {
    hint: 'Length, color, texture, parting and how it falls. LoRA-safe still describes the hair.',
    instr: 'Extract the hair only.\n\n' +
      'Cover length, color, texture, parting, style (open, bun, braid, pinned), how it falls on the face and shoulders, and any cloth or metal holding it. If the style is cropped, say what you can see.\n' +
      'Write bullets, then one paragraph that starts with "Hair:" for pasting into a prompt. Do not describe the face, outfit or body.'
  },
  'Pose & expression': {
    hint: 'Weight, hands, head and the readable expression. Where the body is, not what body type it is.',
    instr: 'Extract pose and expression only.\n\n' +
      '## Pose\n' +
      '- Standing, seated, leaning or walking, and where the weight sits\n' +
      '- Shoulder line, torso twist, hip angle\n' +
      '- Each arm and each hand: where it is and what it touches. If a hand is cropped or unclear, say so\n' +
      '- Legs and feet, or "cropped"\n' +
      '- Head tilt\n\n' +
      '## Expression\n' +
      '- Brows, eyes, mouth, and the overall read in plain words (calm, mid-laugh, tense)\n' +
      '- Gaze direction\n' +
      '- Whether the mouth is open, closed or speaking\n\n' +
      'Do not describe identity, outfit or the room except when a hand touches something.\n' +
      'End with two pasteable lines: "Pose:" and "Expression:".'
  },
  'Light & color': {
    hint: 'Key, fill, rim, time-of-day cues, and the grade.',
    instr: 'Extract light and color only.\n\n' +
      '## Light\n' +
      '- Key: direction relative to the subject, hard or soft, and the source if it is in frame or obvious (window, lamp, sun)\n' +
      '- Fill and the shadow side\n' +
      '- Rim or hair light\n' +
      '- Color of the light (warm, neutral, cool, mixed)\n' +
      '- Time of day only if the frame shows it\n\n' +
      '## Color\n' +
      '- Dominant palette, in concrete color names\n' +
      '- Saturation and contrast (muted, natural, punched)\n' +
      '- Accent color\n' +
      '- Grade in one clause (clean digital, soft film, sodium night). Do not name a film stock unless the look is unmistakable.\n\n' +
      'End with one line that starts with "Light:" and can be pasted into a prompt.'
  },
  'Set & props': {
    hint: 'The place, the depth behind the subject, and what is in the hands or on the set.',
    instr: 'Extract the location and the objects. Leave identity, clothes and the face out, except to say where the person stands relative to the set.\n\n' +
      '## Place\n- Indoor or outdoor, and the kind of room or street\n- Architecture, furniture, weather, time cues\n- What is sharp and what falls soft\n\n' +
      '## Props\n- Anything in the hands\n- Objects that matter on the set\n- Visible text on signs or objects, quoted exactly, or "none readable"\n\n' +
      'End with one paragraph that starts with "Set:" for pasting into a prompt.'
  },
  'Image description': {
    hint: 'One tight paragraph of the whole frame, then the facts a later prompt must keep.',
    instr: 'Describe the image in one concrete paragraph: who is present by role only (woman, man, no identity essay), clothing, pose, where they stand, the place and the light.\n' +
      'Then a short bullet list titled "Keep" with the facts a later prompt must not drop: framing, position, outfit colors, jewellery, and any readable text.\n' +
      'Do not speculate about a story, a brand, or camera metadata.'
  },
  'Image to prompt': {
    hint: 'One photoreal still prompt: clothes, pose, framing, light and place.',
    instr: 'Turn this image into ONE photorealistic image prompt of 120 to 180 words. Natural prose, no lists, no headings, no quality tags, no camera brand.\n' +
      'Cover the visible outfit, pose, expression in plain words, where the subject sits in the frame, shot size, camera height, angle, lens feel, place and light.\n' +
      'Name each garment once. Do not invent anything off-frame.\n' +
      'If a character LoRA note appears below, leave face structure, hair, skin tone and body type out. Let the prompt carry pose, garments, framing, place and light.'
  },
  'H3 framing': {
    hint: 'A short blocking note and one sentence you can drop into an H3 shot.',
    instr: 'Write a framing extract for a MiniMax H3 shot that should match this still. The still is a held frame, so do not invent a camera move.\n\n' +
      '## Camera\n' +
      'Shot size, height, facing, lens feel, placement. Use: extreme close-up, close-up, medium close-up, medium shot, medium-wide, wide shot, extreme wide; eye level, slightly low, slightly high, overhead; front-facing, three-quarter, side profile, from behind, over the shoulder.\n\n' +
      '## Character position\n' +
      'Where the person is in the frame, how they are turned, and what the crop includes. One or two sentences. If nobody is in frame, say so and place the main object.\n\n' +
      '## H3 line\n' +
      'One sentence, no labels and no timestamps: "framed at <height> in a <shot size>, <facing>, shot on a <50mm or 85mm> lens with compressed perspective and natural proportions, <centered or on the inner third>."\n' +
      'Use 85mm for a close-up or medium close-up, and 50mm for a medium shot. For a wide shot with no readable face, say "seen from behind in a wide shot" or "an extreme wide shot" and do not force an 85mm.\n' +
      'Add one last line, "Hold: the camera holds static."'
  },
  'Edit inventory': {
    hint: 'Current state of each region, plus a Keep sentence for a Qwen edit.',
    instr: 'List what is currently in the image, as short lines a Qwen edit can keep or change. One line each, starting with the label:\n' +
      'Outfit:\nJewellery:\nMakeup:\nHair:\nPose:\nExpression:\nFraming:\nBackground:\nLight:\n\n' +
      'Then a line "Keep:" that names, by role, what an identity-safe edit should leave alone: the person\'s identity, the pose, the framing and the background. Change that line if one of those is missing from the frame.\n' +
      'Do not write the edit itself. If a region is not visible, write "not visible".'
  },
  'Full breakdown': {
    hint: 'Camera, position, outfit, jewellery, makeup, pose, light and set in one pass.',
    instr: 'Extract a compact breakdown. Use these markdown headings, in this order, with short bullets:\n' +
      '## Camera\n## Character position\n## Outfit\n## Jewellery\n## Makeup\n## Pose & expression\n## Light & color\n## Set & props\n## Visible text\n## H3 line\n\n' +
      'Camera uses shot size, height, facing and lens feel (85mm for a close view, 50mm for medium, wide only when faces are small).\n' +
      'Character position says where each person sits, the depth, body orientation and gaze. If nobody is in frame, say so.\n' +
      'The H3 line is one sentence: "framed at <height> in a <shot size>, <facing>, shot on a <lens>, <placement>."\n' +
      'Visible text is quoted exactly, or "none readable".\n' +
      'If a character LoRA note appears below, skip hair, face structure, skin tone and body type. Still include makeup, outfit, jewellery, camera and character position. Position is where the body is, not the body type.'
  }
};
function pickEx(k){
  if (!EX_PRESETS[k]) k = 'Camera & blocking';
  EX_PRESET = k;
  S('extract').preset = k;
  persist();
  drawChips('ex-presets', EX_PRESETS, k, pickEx);
  $('#ex-hint').textContent = EX_PRESETS[k].hint;
  const loraLabel = $('#ex-lora') && $('#ex-lora').parentElement;
  if (loraLabel && loraLabel.lastChild) {
    loraLabel.lastChild.textContent = k === 'Hair'
      ? ' LoRA-safe — leave face, skin and body type out'
      : ' LoRA-safe — leave face, hair, skin and body type out';
  }
}
function drawEx(){ drawThumbs('ex-thumbs', EX_IMG ? [EX_IMG] : [], () => { EX_IMG = null; drawEx(); }); }
function exInstruction(){
  let content = EX_PRESETS[EX_PRESET].instr;
  const focus = $('#ex-focus').value.trim();
  if (focus) content += '\n\nRead only this: ' + focus + '. If they are not in the frame, say so and stop.';
  if ($('#ex-lora').checked && EX_PRESET === 'Hair') {
    content += '\n\nA character LoRA supplies face structure, skin tone and body type. Describe the hair. Leave face structure, skin tone and body type out.';
  } else if ($('#ex-lora').checked) {
    content += '\n\nA character LoRA supplies face structure, hair, skin tone and body type. Leave those out. Still report position, orientation, crop, garments, jewellery, makeup, pose and light when this extract asks for them. Position is where the body is, not the body type.';
  }
  const also = $('#ex-also').value.trim();
  if (also) content += '\n\nAlso include: ' + also;
  content += '\n\nUse only what is visible. If a detail is cropped, hidden or unclear, say "not visible" or "unclear". Do not invent brands, readable text or jewellery.';
  return content;
}
setupDrop('ex-drop', 'ex-file', async files => {
  const imgs = await rememberPaste(files);
  if (!imgs[0]) { ST.extract.status.textContent = 'That file is not an image.'; return; }
  EX_IMG = imgs[0].b64;
  drawEx();
  if (imgs.length > 1) toast('Extract uses the first image');
});
$('#ex-focus').oninput = e => { S('extract').focus = e.target.value; persist(); };
$('#ex-also').oninput = e => { S('extract').also = e.target.value; persist(); };
$('#ex-lora').onchange = e => { S('extract').loraSafe = e.target.checked; persist(); };
$('#ex-clear').onclick = () => { EX_IMG = null; drawEx(); };
$('#ex-run').onclick = () => busy($('#ex-run'), async () => {
  if (!EX_IMG) { ST.extract.status.textContent = 'Add an image first.'; return; }
  const out = await runInto('extract', withSys('extract', [{role:'user', content: exInstruction(), images:[EX_IMG]}]));
  if (ST.extract.last === 'ok' && out.trim()) await remember('extract', EX_PRESET, out, {model: modelOf('extract'), preset: EX_PRESET});
});


// ---------- Perchance generators: shared helpers ----------
function pcOpts(sel, raw, cur){
  const el = typeof sel === 'string' ? $(sel) : sel;
  el.innerHTML = String(raw || '').split('|||').filter(Boolean).map(p => { const i = p.indexOf('='); const v = i < 0 ? p : p.slice(0, i), l = i < 0 ? p : p.slice(i + 1); return `<option value="${esc(v)}">${esc(l)}</option>`; }).join('');
  if (cur != null && [...el.options].some(o => o.value === cur)) el.value = cur;
}
function keepFields(tab){
  $$('#tab-' + tab + ' [data-keep]').forEach(el => {
    const s = S(tab), k = el.id;
    if (k in s) { if (el.type === 'checkbox') el.checked = !!s[k]; else if (el.tagName !== 'SELECT' || [...el.options].some(o => o.value === s[k])) el.value = s[k]; }
    const save = () => { S(tab)[k] = el.type === 'checkbox' ? el.checked : el.value; persist(); };
    el.addEventListener('input', save); el.addEventListener('change', save);
  });
}
function setKeep(id, v){ const el = $('#' + id); if (!el) return; if (el.type === 'checkbox') el.checked = !!v; else el.value = v == null ? '' : v; el.dispatchEvent(new Event('change')); }
async function pcAsk(tab, content, imgs){
  const msg = {role:'user', content};
  if (imgs && imgs.length) msg.images = imgs;
  const out = await runInto(tab, withSys(tab, [msg]));
  return ST[tab].last === 'err' ? '' : out;
}
function pcClean(t){ return String(t || '').replace(/^\s*```\w*\s*/,'').replace(/```\s*$/,'').trim(); }

// ---------- T2VA ----------
const T2_TYPES = [['auto','auto (random type)'],['hollywoodSilent','Hollywood silent 1920s'],['hollywoodGolden','Hollywood golden 1930s–40s'],['hollywoodNoir','Hollywood noir 1940s–50s'],['hollywood1970s','Hollywood 1970s'],['hollywood1980s','Hollywood 1980s'],['hollywood1990s','Hollywood 1990s'],['hollywoodNow','Hollywood now'],['bollywoodClassic','Bollywood 1950s–60s'],['bollywood1970s','Bollywood 1970s'],['bollywood1980s','Bollywood 1980s'],['bollywood1990s','Bollywood 1990s'],['bollywoodNow','Bollywood now'],['anime','Anime'],['cartoon','Cartoon'],['commercial','TV commercial'],['musicVideo','Music video'],['soap','TV soap'],['horror','Horror'],['documentary','Documentary'],['exploitation','Exploitation']];
$('#t2-type').innerHTML = T2_TYPES.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
const T2_LOC = {lside:'locSide', lera:'locEra', ltime:'locTime', lweather:'locWeather', lmood:'locMood', lstyle:'locStyle'};
function t2AnyOpts(el, list){ el.innerHTML = '<option value="">Any</option>' + list.map(x => `<option>${esc(x)}</option>`).join(''); }
function t2Places(){ t2AnyOpts($('#t2-lplace'), $('#t2-lcat').value ? PC_T2VA.locOptions('locPlace', $('#t2-lcat').value).split('|||') : []); }
t2AnyOpts($('#t2-lcat'), PC_T2VA.locOptions('locPlace', '').split('|||'));
$('#t2-lcat').onchange = t2Places; t2Places();
Object.entries(T2_LOC).forEach(([id, name]) => t2AnyOpts($('#t2-' + id), PC_T2VA.locOptions(name, '').split('|||')));
function t2Pick(el){ const o = [...el.options].filter(x => x.value); return el.value || (o.length ? o[Math.floor(Math.random() * o.length)].value : ''); }
function t2Spec(){
  const cat = t2Pick($('#t2-lcat'));
  let place = $('#t2-lcat').value ? t2Pick($('#t2-lplace')) : '';
  if (!place && cat) place = String(PC_T2VA.locPlace[cat] || '');
  const s = {place, category: cat};
  Object.keys(T2_LOC).forEach(id => { s[id.slice(1)] = t2Pick($('#t2-' + id)); });
  return s;
}
function t2State(){ const s = S('t2va'); s.scenes = Array.isArray(s.scenes) ? s.scenes : []; return s; }
function t2Type(forStart){
  const v = $('#t2-type').value, st = t2State();
  if (v !== 'auto') return v;
  if (!forStart && st.scenes.length) return st.scenes[st.scenes.length - 1].type || 'hollywoodNow';
  return String(PC_T2VA.randomTypeName() || 'hollywoodNow');
}
function t2Bible(){ return $('#t2-bible').value.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.startsWith('- ') ? l : '- ' + l).join('\n'); }
function t2Names(){ return $('#t2-bible').value.split('\n').map(l => l.split(/:|\|\|/)[0].replace(/^- /, '').trim()).filter(Boolean).join(', '); }
function t2Reel(){ const st = t2State(); $('#t2-reel').textContent = st.scenes.length ? 'Reel: ' + st.scenes.length + ' clip(s) · ' + st.scenes.map((c, i) => (i + 1) + '. ' + (T2_TYPES.find(t => t[0] === c.type) || [0, c.type])[1]).join(' · ') : 'Reel is empty. Next scene starts clip 1.'; }
async function t2Make(mode){
  const st = t2State(), dur = $('#t2-dur').value, user = $('#t2-dir').value;
  let type = t2Type(mode !== 'next' || !st.scenes.length), prev = '', n = 1, spark = '';
  if (mode === 'next' && st.scenes.length) { prev = String(st.scenes[st.scenes.length - 1].text || '').slice(-3500); n = st.scenes.length + 1; }
  else if (mode === 'next') { mode = 'start'; }
  if (mode === 'start') spark = String(PC_T2VA.randomSpark(type));
  if (mode === 'generate') st.scenes = [];
  const kind = mode === 'generate' ? 'generate' : (n > 1 ? 'next' : 'start');
  ST.t2va.label = (n > 1 ? 'Scene ' + n : 'Opening') + ' · ' + type;
  let text = pcClean(await pcAsk('t2va', PC_T2VA.buildT2va(kind, user, dur, type, prev, n, spark, t2Bible(), $('#t2-loc').value.trim())));
  if (!text || !/integrated_multimodal_description/i.test(text)) {
    if (!text) { text = PC_T2VA.fillInT2va(n > 1 ? 'next' : 'start', user, dur, type, prev, t2Bible(), $('#t2-loc').value.trim()); paintOut('t2va', text); ST.t2va.status.textContent = 'AI failed or returned a bad block. Fill-in used.'; }
  }
  st.scenes.push({type, text, dur}); if (st.scenes.length > 30) st.scenes = st.scenes.slice(-30); persist(); t2Reel();
  await remember('t2va', 'Clip ' + st.scenes.length + ' · ' + type, text, {model: modelOf('t2va')});
}
$('#t2-next').onclick = () => busy($('#t2-next'), () => t2Make('next'));
$('#t2-gen').onclick = () => busy($('#t2-gen'), () => t2Make('generate'));
$('#t2-reset').onclick = () => { t2State().scenes = []; persist(); t2Reel(); toast('Reel cleared'); };
$('#t2-bible-clear').onclick = () => setKeep('t2-bible', '');
$('#t2-lock').onclick = () => busy($('#t2-lock'), async () => {
  const st = t2State(), clip = ST.t2va.out.value.trim() || (st.scenes.length ? st.scenes[st.scenes.length - 1].text : '');
  if (!clip) { ST.t2va.status.textContent = 'Generate a scene first.'; return; }
  const saved = ST.t2va.out.value;
  const raw = await pcAsk('t2va', PC_T2VA.bibleExtract(clip, t2Bible()));
  paintOut('t2va', saved);
  const have = new Set(t2Names().toLowerCase().split(', '));
  const add = raw.split('\n').map(l => l.replace(/\*\*/g, '').replace(/^[-*\d.)\s]+/, '').trim()).filter(l => l.includes('||')).map(l => { const [a, ...b] = l.split('||'); return [a.trim(), b.join('||').trim()]; }).filter(([a, b]) => a && b && !have.has(a.toLowerCase()));
  if (!add.length) { ST.t2va.status.textContent = 'No new characters found.'; return; }
  setKeep('t2-bible', ($('#t2-bible').value.trim() + '\n' + add.map(([a, b]) => a + ': ' + b).join('\n')).trim());
  ST.t2va.status.textContent = 'Locked ' + add.length + ' character(s).';
});
async function t2LocAsk(instr){
  const saved = ST.t2va.out.value;
  const raw = pcClean(await pcAsk('t2va', instr));
  paintOut('t2va', saved);
  const line = raw.split('\n').map(l => l.replace(/\*\*/g, '').trim()).find(l => l.includes('||')) || raw.split('\n')[0] || '';
  if (line) setKeep('t2-loc', line);
}
$('#t2-lgen').onclick = () => busy($('#t2-lgen'), () => t2LocAsk(PC_T2VA.locationWrite($('#t2-lidea').value.trim() || PC_T2VA.locSpecText(t2Spec()), t2Type(false), t2Names())));
$('#t2-lrand').onclick = () => busy($('#t2-lrand'), () => t2LocAsk(PC_T2VA.locationRandom(t2Spec(), t2Type(false), t2Names())));
$('#t2-lenh').onclick = () => busy($('#t2-lenh'), () => { const [n, ...d] = $('#t2-loc').value.split('||'); return t2LocAsk(PC_T2VA.locationEnhance(n.trim(), d.join('||').trim(), t2Type(false))); });
$('#t2-linstant').onclick = () => setKeep('t2-loc', String(PC_T2VA.locInstant(t2Spec())));
$('#t2-lroll').onclick = () => { ['lcat', ...Object.keys(T2_LOC)].forEach(id => { const el = $('#t2-' + id); el.value = t2Pick(el); }); t2Places(); $('#t2-lplace').value = t2Pick($('#t2-lplace')); };
$('#t2-lany').onclick = () => { ['lcat', 'lplace', ...Object.keys(T2_LOC)].forEach(id => { $('#t2-' + id).value = ''; }); t2Places(); };

// ---------- Ref2VA ----------
function r2Cast(){
  const rows = id => $('#' + id).value.split('\n').map(l => l.trim()).filter(Boolean).map(l => l.split('|').map(x => x.trim()));
  return JSON.stringify({
    people: rows('r2-people').map(([name, kind, ...note]) => ({name: name || '', kind: (kind || 'person').toLowerCase(), note: note.join(' | ')})),
    places: rows('r2-places').map(([name]) => ({name})),
    audios: rows('r2-audios').map(([label, p, ...t]) => ({label: label || '', personIndex: Math.max(0, (parseInt(p, 10) || 1) - 1), timbre: t.join(' | ')}))
  });
}
function r2Budget(){ const c = JSON.parse(r2Cast()), imgs = c.people.length + c.places.length; $('#r2-budget').textContent = 'Images: ' + imgs + ' / 9 · Voices: ' + c.audios.length + ' / 3' + (imgs > 9 || c.audios.length > 3 ? ' — extra entries are dropped' : ''); }
['r2-people','r2-places','r2-audios'].forEach(id => $('#' + id).addEventListener('input', r2Budget));
$('#r2-run').onclick = () => busy($('#r2-run'), async () => {
  const cast = r2Cast(), text = $('#r2-text').value, dur = $('#r2-dur').value, lang = $('#r2-lang').value;
  let out = pcClean(await pcAsk('ref2va', PC_REF.buildRef2va(cast, text, dur, lang)));
  if (!out) { out = PC_REF.fillInRef2va(cast, text, dur, lang); paintOut('ref2va', out); ST.ref2va.status.textContent = 'AI failed. Fill-in used.'; }
  await remember('ref2va', text.slice(0, 60) || 'Ref2VA', out, {model: modelOf('ref2va')});
});
$('#r2-fill').onclick = () => paintOut('ref2va', PC_REF.fillInRef2va(r2Cast(), $('#r2-text').value, $('#r2-dur').value, $('#r2-lang').value));

// ---------- Soundscape ----------
const SD_SEL = {'sd-music':'musicTypeOptions','sd-genre':'genreOptions','sd-tempo':'tempoOptions','sd-era':'eraOptions','sd-vocal':'vocalOptions','sd-lang':'langOptions','sd-sound':'soundOptions'};
Object.entries(SD_SEL).forEach(([id, fn]) => pcOpts('#' + id, PC_SOUND[fn]()));
$$('#sd-durs button').forEach(b => b.onclick = () => setKeep('sd-dur', b.dataset.dur));
function sdArgs(){ return [$('#sd-text').value, $('#sd-sound').value, $('#sd-music').value, $('#sd-genre').value, $('#sd-tempo').value, $('#sd-era').value, $('#sd-vocal').value, $('#sd-lang').value, Math.max(5, Math.min(600, +$('#sd-dur').value || 10))]; }
$('#sd-run').onclick = () => busy($('#sd-run'), async () => {
  const a = sdArgs();
  let out = pcClean(await pcAsk('sound', PC_SOUND.buildSound(...a)));
  if (!out) { out = PC_SOUND.fillInSound(...a); paintOut('sound', out); ST.sound.status.textContent = 'AI failed. Fill-in used.'; }
  await remember('sound', a[0].slice(0, 60) || 'Soundscape', out, {model: modelOf('sound')});
});
$('#sd-fill').onclick = () => paintOut('sound', PC_SOUND.fillInSound(...sdArgs()));
$('#sd-example').onclick = () => { if ($('#sd-text').value.trim()) { toast('Scene already filled'); return; } setKeep('sd-text', 'Night. A wet Bombay street. She steps out of a cab in a chiffon sari and looks up at the pink house. He watches from the balcony.'); };

// ---------- Center Orbit ----------
const OR_KEYS = ['style','place','light','sound','north','east','south','west','ne','nw','se','sw','middle'];
let OR_IMG = null;
function orRead(){ const o = {dur: $('#or-dur').value, where: $('#or-where').value}; OR_KEYS.forEach(k => { o[k] = $('#or-' + k).value; }); return o; }
function orBuild(kind){ const o = orRead(); return PC_ORBIT.buildOrbit(kind, o.dur, o.style, o.place, o.light, o.sound, o.north, o.east, o.south, o.west, o.ne, o.nw, o.se, o.sw, o.middle, o.where); }
function orNotes(){ const o = orRead(); $('#or-notes').textContent = String(PC_ORBIT.surveyNotes(o.north, o.east, o.south, o.west, o.ne, o.nw, o.se, o.sw, o.middle) || '').split('\n').filter(Boolean).join(' · '); }
function orApply(raw){
  const t = pcClean(raw); let d = null;
  try { d = JSON.parse(t); } catch (e) { const m = t.match(/\{[\s\S]*\}/); if (m) { try { d = JSON.parse(m[0]); } catch (e2) {} } }
  if (!d) { ST.orbit.status.textContent = 'Could not read that as room data. Try again with one more concrete object.'; return; }
  OR_KEYS.forEach(k => { if (d[k] != null) setKeep('or-' + k, String(d[k]).replace(/\s+/g, ' ').trim()); });
  if (['above','floor','waist'].includes(d.where)) setKeep('or-where', d.where);
  const q = Array.isArray(d.questions) ? d.questions.filter(Boolean).slice(0, 3) : [];
  ST.orbit.status.textContent = 'Filled the form.' + (q.length ? ' Missing: ' + q.join(' ') : ' Check the walls, then Generate sweep.');
  orNotes();
}
const OR_RULES = 'HARD LAYOUT RULE: a TV and a couch/sofa go on OPPOSITE walls (north<>south or east<>west), never the same or adjacent walls; the couch faces the TV. Every wall phrase starts a/an/one/the, under 14 words, with color AND material. No people, faces, text, or watermarks. Small plain room, no alignment line. Infer style as \'Live-action, YEAR or ERA, PLACE, film stock or look, grain\', place as one sentence with size, light as a short phrase naming the light source, sound as 1-2 sentences of room tone ending \'No voices.\', where as one of above, floor, waist for where MIDDLE sits. Corners ne/nw/se/sw only if a corner object is named or visible, else empty string. MIDDLE = object nearest the camera or hanging above it, else empty string.';
$('#or-interpret').onclick = () => busy($('#or-interpret'), async () => {
  const text = $('#or-desc').value.trim();
  if (!text) { ST.orbit.status.textContent = 'Describe the room first — even one sentence is enough.'; return; }
  orApply(await pcAsk('orbit', 'You turn a freeform room description into a 360-degree room prompt form. The form drives a slow centre-orbit video prompt, so every object needs its own color, material, and shape. Place the most-mentioned / largest object on NORTH, then go clockwise. Fill every wall. ' + OR_RULES + ' Also return questions: up to 3 short questions about real gaps that change the layout, or an empty array.\n\nUser description:\n' + text.slice(0, 2000) + '\n\nReturn ONLY a raw JSON object, no markdown, with exactly these keys: style, place, light, sound, north, east, south, west, ne, nw, se, sw, middle, where, questions (array of strings).'));
});
setupDrop('or-drop', 'or-file', async files => { const imgs = await rememberPaste(files); if (imgs[0]) { OR_IMG = imgs[0].b64; drawThumbs('or-thumbs', [OR_IMG], () => { OR_IMG = null; drawThumbs('or-thumbs', [], () => {}); }); } });
$('#or-autofill').onclick = () => busy($('#or-autofill'), async () => {
  if (!OR_IMG) { ST.orbit.status.textContent = 'Pick a room photo first (use a vision model).'; return; }
  orApply(await pcAsk('orbit', 'You fill a 360-degree room prompt form from ONE room photo (attached). NORTH = the wall or large object in the centre of the photo. WEST = the left edge. EAST = the right edge. SOUTH = your best guess at the unseen opposite wall, in the same style (start with \'same room, opposite wall, \'). ' + OR_RULES + '\n\nReturn ONLY a raw JSON object with exactly these string keys, no markdown: style, place, light, sound, north, east, south, west, ne, nw, se, sw, middle, where.', [OR_IMG]));
});
$('#or-clear').onclick = () => { if (!confirm('Clear the room form?')) return; OR_KEYS.forEach(k => setKeep('or-' + k, '')); orNotes(); };
$('#or-template').onclick = () => {
  const t = {style:'Live-action, 1986 Bombay, Eastman Color, fine film grain', place:'A small creek hut on the Gorai shore, about 3 by 3 metres.', light:'Kerosene light from the hurricane lamp only', sound:'A faint kerosene hiss, creek water, mangrove insects. No voices.', north:'a faded red cloth door, tied open at one side', east:'a rope charpoy with a blue-and-white striped mattress', south:'a pale brown mud wall', west:'one small square window with a wooden shutter', ne:'', nw:'', se:'', sw:'a black tin trunk with brass corners', middle:'one glass hurricane lamp with a brass base'};
  OR_KEYS.forEach(k => setKeep('or-' + k, t[k])); setKeep('or-where', 'above'); orNotes();
};
OR_KEYS.forEach(k => $('#or-' + k).addEventListener('change', orNotes));
$('#or-run').onclick = async () => { const t = orBuild('sweep'); paintOut('orbit', t); ST.orbit.status.textContent = 'Sweep built locally.'; orNotes(); await remember('orbit', 'Sweep · ' + ($('#or-place').value.slice(0, 50) || 'room'), t, {}); };
$('#or-six').onclick = async () => {
  const names = {north:'North', east:'East', south:'South', west:'West', up:'Up', down:'Down'};
  const t = Object.keys(names).map(k => '=== ' + names[k] + ' ===\n' + orBuild(k)).join('\n\n');
  paintOut('orbit', t); ST.orbit.status.textContent = 'Six angles built locally.'; orNotes();
  await remember('orbit', 'Six angles · ' + ($('#or-place').value.slice(0, 50) || 'room'), t, {});
};

// ---------- Character ----------
$('#cg-genre').innerHTML = PC_CHARUI.GENRES.map(([v, l]) => `<option value="${v}">${esc(l)}</option>`).join('');
$('#cg-fields').innerHTML = PC_CHARUI.FIELDS.map(f => `<label${f.long ? ' style="grid-column:1/-1"' : ''}>${esc(f.label)} <textarea id="cg-f-${f.k}" data-keep placeholder="${esc(f.ph)} (or leave empty for AI)"></textarea></label>`).join('');
$('#cg-regen-key').innerHTML = [['name','Name'],['role','Role / archetype'], ...PC_CHARUI.FIELDS.map(f => [f.k, f.label])].map(([k, l]) => `<option value="${k}">${esc(l)}</option>`).join('');
function cgEl(k){ return k === 'name' ? $('#cg-name') : k === 'role' ? $('#cg-role') : $('#cg-f-' + k); }
function cgData(){ const d = {name: $('#cg-name').value, role: $('#cg-role').value, genre: $('#cg-genre').value, seed: $('#cg-seed').value, f: {}, refMode: $('#cg-ref').checked}; PC_CHARUI.FKEYS.forEach(k => { d.f[k] = $('#cg-f-' + k).value; }); return d; }
async function cgOutputs(){
  const raw = await pcAsk('character', PC_CHARUI.outputsPrompt(cgData()));
  const got = PC_CHARUI.parseFields(raw, ['visual', 'voice'], PC_CHARUI.OUT_ALIASES);
  const d = cgData();
  const visual = String(got.visual || PC_CHARUI.fallbackVisual(d)).replace(/\s+/g, ' ').trim();
  setKeep('cg-visual', visual);
  setKeep('cg-voice', String(got.voice || d.f.voice || '').replace(/\s+/g, ' ').trim());
  setKeep('cg-sheet', PC_CHARUI.buildSheet(d.name, visual, d.genre, d.refMode));
  if (!got.visual) ST.character.status.textContent = 'AI visual line was unusable; built one from your fields instead.';
  await remember('character', (d.name || 'Character') + ' · outputs', 'Visual: ' + visual + '\n\nVoice: ' + $('#cg-voice').value + '\n\nSheet:\n' + $('#cg-sheet').value, {model: modelOf('character')});
}
$('#cg-dice').onclick = () => { const n = PC_CHAR.seeds[$('#cg-genre').value]; setKeep('cg-seed', String(n || PC_CHAR.seeds)); };
$('#cg-new').onclick = () => { if (!confirm('Clear this character?')) return; ['name','role','seed','visual','sheet','voice'].forEach(k => setKeep('cg-' + k, '')); PC_CHARUI.FKEYS.forEach(k => setKeep('cg-f-' + k, '')); };
$('#cg-develop').onclick = () => busy($('#cg-develop'), async () => {
  if (!$('#cg-seed').value.trim() && !PC_CHARUI.FILLABLE.some(k => cgEl(k).value.trim())) setKeep('cg-seed', String(PC_CHAR.seeds[$('#cg-genre').value] || ''));
  const missing = PC_CHARUI.FILLABLE.filter(k => !cgEl(k).value.trim());
  if (missing.length) {
    const got = PC_CHARUI.parseFields(await pcAsk('character', PC_CHARUI.developPrompt(cgData(), missing)), missing);
    missing.forEach(k => { if (got[k] && !cgEl(k).value.trim()) setKeep(cgEl(k).id, got[k]); });
    if (ST.character.last === 'err') return;
  }
  await cgOutputs();
});
$('#cg-deepen').onclick = () => busy($('#cg-deepen'), async () => {
  const got = PC_CHARUI.parseFields(await pcAsk('character', PC_CHARUI.deepenPrompt(cgData())), ['backstory', 'arc']);
  ['backstory', 'arc'].forEach(k => { if (got[k]) setKeep('cg-f-' + k, got[k]); });
});
$('#cg-build').onclick = () => busy($('#cg-build'), cgOutputs);
$('#cg-regen').onclick = () => busy($('#cg-regen'), async () => {
  const k = $('#cg-regen-key').value;
  const got = PC_CHARUI.parseFields(await pcAsk('character', PC_CHARUI.regenPrompt(cgData(), k)), [k]);
  if (got[k]) setKeep(cgEl(k).id, got[k]); else ST.character.status.textContent = 'Could not read a new ' + k + '.';
});
$('#cg-to-t2va').onclick = () => {
  const v = $('#cg-visual').value.trim(); if (!v) { toast('Build outputs first'); return; }
  setKeep('t2-bible', ($('#t2-bible').value.trim() + '\n' + ($('#cg-name').value.trim() || 'Unnamed') + ': ' + v).trim());
  showTab('t2va'); toast('Locked in T2VA');
};
// ---- Enhance / Generate (prompts live in ASSIST_PROMPTS in Python, served as CFG.assist) ----
const AS_UNDO = {};
function asCfg(tab){ return (CFG.assist || {})[tab]; }
function asMap(c){ return typeof c.fields === 'string' ? null : c.fields; }
function asIds(c){ return typeof c.fields === 'string' ? [c.fields] : Object.values(c.fields); }
function asLabelOf(el){ const l = el.closest('label'); return l ? (l.childNodes[0] && l.childNodes[0].textContent || l.textContent).trim().replace(/\s+/g, ' ').slice(0, 60) : (el.id || ''); }
function asContext(tab, c){
  const skip = new Set(asIds(c).concat(c.idea ? [c.idea] : []));
  const out = [];
  $$(`#tab-${tab} select, #tab-${tab} input[type=text], #tab-${tab} input[type=number], #tab-${tab} input:not([type]), #tab-${tab} textarea`).forEach(el => {
    if (skip.has(el.id) || !el.closest('label') || /^send to/i.test(asLabelOf(el)) || el.closest('.modelbar') || el.closest('.assist') || el.type === 'file') return;
    let v = el.tagName === 'SELECT' ? (el.selectedOptions[0] ? el.selectedOptions[0].textContent : '') : el.value;
    v = String(v || '').trim();
    if (!v) return;
    out.push(asLabelOf(el) + ': ' + (v.length > 1500 ? v.slice(0, 1500) + '…' : v));
  });
  return out.join('\n');
}
function asStrip(t){
  t = String(t || '').replace(/<think>[\s\S]*?<\/think>/g, '');
  t = pcClean(t);
  t = t.replace(/^\s*(sure|okay|ok|certainly|here(?: is|'s| are)[^\n]*?)[:!.]\s*\n/i, '');
  t = t.replace(/^\s*(enhanced|improved|rewritten|generated|final)[^\n:]{0,30}:\s*\n?/i, '');
  return t.trim().replace(/^"([\s\S]*)"$/, '$1').trim();
}
function asParse(text, keys){
  const out = {}, re = new RegExp('^\\s*[-*#>]*\\s*\\**\\s*(' + keys.join('|') + ')\\s*\\**\\s*[:=]\\s*\\**', 'i');
  let cur = null;
  String(text || '').split(/\r?\n/).forEach(line => {
    const m = line.match(re);
    if (m) { cur = keys.find(k => k.toLowerCase() === m[1].toLowerCase()); out[cur] = line.slice(m[0].length).trim(); }
    else if (cur) out[cur] += '\n' + line;
  });
  for (const k in out) out[k] = out[k].trim().replace(/^(empty|none|n\/a|-)$/i, '');
  return out;
}
function asUndoBtn(tab){ return $(`.assist[data-for="${tab}"] .as-undo`); }
async function asRun(tab, mode, btn){
  const c = asCfg(tab); if (!c) return;
  const map = asMap(c), ids = asIds(c);
  const snap = {}; ids.forEach(id => { const el = $('#' + id); if (el) snap[id] = el.value; });
  const ideaId = c.idea || ids[0];
  const ideaVal = (($('#' + ideaId) || {}).value || '').trim();
  let body = (mode === 'enhance' ? c.enhance : c.generate) + '\n\n';
  if (map) {
    const keys = Object.keys(map);
    if (mode === 'enhance') body += 'Current values:\n' + keys.map(k => k + ': ' + (snap[map[k]] || '').trim()).join('\n') + '\n\n';
    else body += 'Idea: ' + (ideaVal || '(none — invent one; variation ' + Math.floor(Math.random() * 1e4) + ')') + '\n\n';
    body += 'Answer with exactly these lines, in this order, each starting with the key and a colon (leave a value empty if it should be empty):\n' + keys.map(k => k + ': ...').join('\n');
  } else {
    const cur = (snap[ids[0]] || '').trim();
    if (mode === 'enhance') body += '<input>\n' + cur + '\n</input>';
    else body += 'Idea: ' + (cur || '(none — invent one; variation ' + Math.floor(Math.random() * 1e4) + ')');
    body += '\n\nAnswer with the finished text only.';
  }
  const ctx = asContext(tab, c);
  if (ctx) body = 'Tool settings (use them, do not repeat them):\n' + ctx + '\n\n' + body;
  const msgs = [{role:'system', content: c.system + (CFG.assistRules || '')}, {role:'user', content: body}];
  const lbl = ST[tab] ? ST[tab].label : '';
  if (ST[tab]) ST[tab].label = mode === 'enhance' ? '✨ Enhance' : '🎲 Generate';
  const single = !map && $('#' + ids[0]);
  let raw = '';
  try {
    raw = await streamChat(tab, msgs, t => { if (single) single.value = asStrip(t); });
  } finally { if (ST[tab]) ST[tab].label = lbl; }
  if (!ST[tab] || ST[tab].last === 'err' || !String(raw || '').trim()) {
    ids.forEach(id => { if (id in snap) setKeep(id, snap[id]); });
    if (ST[tab] && ST[tab].last !== 'abort') toast('Nothing came back — try again or pick another model.');
    return;
  }
  if (map) {
    const got = asParse(asStrip(raw), Object.keys(map));
    if (!Object.keys(got).length) { toast('The model did not use the KEY: value format — try again or a bigger model.'); return; }
    for (const k in got) setKeep(map[k], got[k]);
  } else setKeep(ids[0], asStrip(raw));
  (AS_UNDO[tab] = AS_UNDO[tab] || []).push(snap);
  const u = asUndoBtn(tab); if (u) u.disabled = false;
  if (ST[tab]) ST[tab].status.textContent = (mode === 'enhance' ? 'Enhanced' : 'Generated') + ' with ' + modelOf(tab) + ' · ↶ Undo restores the previous text';
}
function asUndo(tab){
  const s = (AS_UNDO[tab] || []).pop(); if (!s) return;
  for (const id in s) setKeep(id, s[id]);
  const u = asUndoBtn(tab); if (u) u.disabled = !(AS_UNDO[tab] || []).length;
  toast('Restored');
}
function asBoot(){
  for (const tab of Object.keys(CFG.assist || {})) {
    const c = asCfg(tab), first = $('#' + (c.idea || asIds(c)[0]));
    const panel = $('#tab-' + tab);
    if (!first || !panel || $(`.assist[data-for="${tab}"]`)) continue;
    const anchor = first.closest('label') || first;
    const what = asMap(c) ? 'all ' + Object.keys(asMap(c)).length + ' fields below' : asLabelOf(first).toLowerCase();
    const bar = document.createElement('div');
    bar.className = 'assist'; bar.dataset.for = tab;
    bar.innerHTML = `<button type="button" class="as-enh" title="Rewrite what you typed into a richer, well-structured version that keeps your intent">✨ Enhance</button>`
      + `<button type="button" class="as-gen" title="Write a complete input from a short idea, or from scratch if empty">🎲 Generate</button>`
      + `<button type="button" class="as-undo" title="Restore the text from before the last Enhance / Generate" disabled>↶ Undo</button>`
      + `<span class="hint">Works on ${esc(what)} · uses this tab's model</span>`;
    anchor.parentNode.insertBefore(bar, anchor);
    $('.as-enh', bar).onclick = e => busy(e.currentTarget, () => asRun(tab, 'enhance'));
    $('.as-gen', bar).onclick = e => busy(e.currentTarget, () => asRun(tab, 'generate'));
    $('.as-undo', bar).onclick = () => asUndo(tab);
  }
}
function pcBoot(){
  ['t2va','ref2va','sound','orbit','character'].forEach(keepFields);
  const lc = S('t2va')['t2-lcat']; if (lc != null) { $('#t2-lcat').value = lc; t2Places(); if (S('t2va')['t2-lplace'] != null) $('#t2-lplace').value = S('t2va')['t2-lplace']; }
  t2Reel(); r2Budget(); orNotes();
}

(async function boot(){
  try {
    try { CFG = await (await fetch('/api/config')).json(); } catch (e) {}
    CFG.settings = CFG.settings || {};
    CFG.languages = CFG.languages || [];
    CFG.systems = CFG.systems || {};
    CFG.defaults = CFG.defaults || {};
    const local = JSON.parse(localStorage.getItem('wiz-studio-settings') || '{}');
    CFG.settings = Object.assign({}, local, CFG.settings);
    applyTheme();
    buildOutputs();
    ensureRuntime();
    buildModelBars();
    fillLangs();
    drawOcrChips();
    $('#ocr-instr').value = S('ocr').instr || 'Extract all text.';
    $('#ocr-lang').value = S('ocr').lang || '';
    $('#ocr-pdf-hint').textContent = CFG.pdf ? 'PDFs are split into pages (up to 30), one request per page. Paste works on this tab.' : 'PDF support needs pdftoppm (poppler-utils).';
    $('#tr-tone').value = S('translate').tone || 'neutral';
    if (typeof S('translate').breaks === 'boolean') $('#tr-breaks').checked = S('translate').breaks;
    $('#tr-gloss').value = S('translate').glossary || '';
    const preset = PG_LEGACY[S('prompt').preset] || S('prompt').preset;
    pickPg(preset in PG_PRESETS ? preset : 'Photoreal still');
    if (S('prompt').count) $('#pg-count').value = S('prompt').count;
    if (S('prompt').aspect) $('#pg-aspect').value = S('prompt').aspect;
    $('#pg-trigger').value = S('prompt').trigger ?? 'suh4na woman';
    $('#pg-lora').checked = !!S('prompt').lora;
    rwFill();
    if (S('rewrite').strength) $('#rw-strength').value = S('rewrite').strength;
    if (!$('#rw-strength').value) $('#rw-strength').value = 'medium';
    $('#rw-extra').value = S('rewrite').custom || '';
    pickRw(S('rewrite').preset || 'paraphrase');
    pcBoot();
    asBoot();
    $('#rp-user').value = S('roleplay').user || 'Roshan';
    $('#cap-trigger').value = S('caption').trigger ?? S('prompt').trigger ?? 'suh4na woman';
    pickCap(S('caption').preset in CAP_PRESETS ? S('caption').preset : 'Training');
    drawCaps();
    $('#ex-focus').value = S('extract').focus || '';
    $('#ex-also').value = S('extract').also || '';
    if (typeof S('extract').loraSafe === 'boolean') $('#ex-lora').checked = S('extract').loraSafe;
    pickEx(S('extract').preset in EX_PRESETS ? S('extract').preset : 'Camera & blocking');
    drawEx();
    const hashTab = location.hash.slice(1);
    showTab(TABS.includes(hashTab) ? hashTab : (S('_ui').tab || 'ocr'));
    await loadModels();
    if (rwExtra(RW_PRESET) && rwExtra(RW_PRESET).model) setModel('rewrite', rwExtra(RW_PRESET).model);
    await loadChars();
    await loadThreads();
    document.documentElement.dataset.booted = '1';
  } catch (e) {
    banner('Studio failed to start: ' + (e && e.message ? e.message : e));
  }
})();
</script>
</body>
</html>
"""


TABS_PY = ("ocr", "translate", "prompt", "roleplay", "rewrite", "chat", "caption", "extract",
           "t2va", "ref2va", "sound", "orbit", "character")


def self_test():
    ok = True
    problems = []

    def check(name, cond):
        nonlocal ok
        print(("PASS " if cond else "FAIL ") + name)
        if not cond:
            ok = False
            problems.append(name)

    r = ollama_models(OLLAMA_HOST)
    print("ollama:", "ok" if r["ok"] else r.get("error"), len(r["models"]), "models")
    if r.get("ok"):
        caps = r.get("capabilities") or {}
        check("vision capabilities", any("vision" in (v or []) for v in caps.values()))
    routed, switched = pick_vision_model("gemma-4-27B-Q4KM:latest", ["gemma-4-7.5B-Q8:latest", "qwen-3-vl-8B-Q8:latest"])
    check("text model rerouted", switched and routed == "qwen-3-vl-8B-Q8:latest")
    kept, kept_switched = pick_vision_model("qwen-3-vl-8B-Q8", ["qwen-3-vl-8B-Q8:latest"])
    check("vision model kept", not kept_switched and kept == "qwen-3-vl-8B-Q8:latest")
    check("theme marker", "/*__THEME__*/" in HTML)
    for marker in ("tab-ocr", "tab-translate", "tab-prompt", "tab-roleplay", "tab-rewrite",
                   "tab-chat", "tab-caption", "tab-extract", "cannot read images", "tab-t2va", "tab-ref2va",
                   "tab-sound", "tab-orbit", "tab-character", "PC_RW", "rw-strength", "Camera & blocking",
                   "Image to prompt", "Full breakdown", 'id="lib-modal"', 'id="keys-modal"',
                   "/api/library", "/api/zip", "/api/threads"):
        check("has " + marker, marker in HTML)
    check("has assist bar", "function asBoot" in HTML and "asBoot();" in HTML and ".assist {" in HTML)
    for tab, c in ASSIST_PROMPTS.items():
        ids = [c["fields"]] if isinstance(c["fields"], str) else list(c["fields"].values())
        ids += [c["idea"]] if c.get("idea") else []
        good = all(isinstance(c.get(k), str) and len(c[k]) > 20 for k in ("system", "enhance", "generate"))
        check("assist " + tab, good and tab in TABS_PY and all(
            ('id="%s"' % i) in HTML or (i.startswith("cg-f-") and '"%s"' % i[5:] in HTML) for i in ids))
    check("assist rules", "No preamble" in ASSIST_RULES and "Keep every" in ASSIST_RULES)
    check("safe id", _safe_id("A b/c") == "A_b_c")
    check("safe id empty", _safe_id("") == "default")
    check("safe name", _safe_name("../x.txt") == "x.txt")
    with tempfile.TemporaryDirectory() as td:
        p = Path(td) / "library.json"
        saved = library_add({"tab": "prompt", "title": "Hello", "text": "world"}, path=p)
        items = library_list(p)
        check("library add", len(items) == 1 and items[0]["text"] == "world" and items[0]["id"] == saved["id"])
        library_delete(saved["id"], path=p)
        check("library delete", library_list(p) == [])
        chars = _clean_characters([{"name": "Mira", "persona": "warm", "notes": "local", "tags": "South Indian"}])
        check("character card", chars[0]["name"] == "Mira" and chars[0]["notes"] == "local" and chars[0]["tags"] == "South Indian")
        check("starter roster", len(DEFAULT_CHARACTERS) >= 6 and all(c.get("name") and c.get("persona") for c in DEFAULT_CHARACTERS))
        merged = merge_starter_characters([{"id": "mira", "name": "Mira", "persona": "custom"}])
        check("merge starters keeps custom", next(c for c in merged if c["id"] == "mira")["persona"] == "custom")
        check("merge starters adds others", len(merged) == len(DEFAULT_CHARACTERS))
        msgs = _clean_messages([{"role": "assistant", "content": "a", "alts": ["a", "b"], "altIndex": 1}])
        check("swipe kept", msgs[0]["content"] == "b" and msgs[0]["altIndex"] == 1)
        cont = _clean_messages([{"role": "user", "content": "*continues*", "continue": True}])
        check("continue flag", cont[0].get("continue") is True)
    raw = base64.b64decode(zip_texts([{"name": "../x.txt", "text": "hi"}, {"name": "x.txt", "text": "two"}]))
    names = zipfile.ZipFile(io.BytesIO(raw)).namelist()
    check("zip names", names == ["x.txt", "x-2.txt"] and raw[:2] == b"PK")
    script = HTML.split("<script>", 1)[1].split("</script>", 1)[0]
    if shutil.which("node"):
        with tempfile.TemporaryDirectory() as td:
            js = Path(td) / "ui.js"
            js.write_text(script, encoding="utf-8")
            proc = subprocess.run(["node", "--check", str(js)], capture_output=True, text=True)
            check("script syntax", proc.returncode == 0)
            if proc.returncode:
                print(proc.stderr[-2000:])
    else:
        print("SKIP script syntax (node not installed)")
    print("SELF-TEST", "PASS" if ok else "FAIL " + " ".join(problems))
    return 0 if ok else 1


def main():
    global OLLAMA_HOST
    ap = argparse.ArgumentParser(description="wizstudio — local Ollama studio")
    ap.add_argument("--host", default="127.0.0.1")
    ap.add_argument("--port", type=int, default=9014)
    ap.add_argument("--ollama", default=OLLAMA_HOST)
    ap.add_argument("--self-test", action="store_true")
    args = ap.parse_args()
    OLLAMA_HOST = args.ollama.rstrip("/")

    if args.self_test:
        sys.exit(self_test())

    for folder in (CONFIG_DIR, HISTORY_DIR, THREAD_DIR):
        folder.mkdir(parents=True, exist_ok=True)
    server = ThreadingHTTPServer((args.host, args.port), Handler)
    print(f"wizstudio: http://{args.host}:{args.port}  (ollama {OLLAMA_HOST})")
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        print("\nstopped")


if __name__ == "__main__":
    main()
