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
            self._send(200, {"defaults": DEFAULT_MODELS, "systems": DEFAULT_SYSTEMS, "languages": LANGUAGES,
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
<link rel="icon" type="image/png" href="data:image/png;base64,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">
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
  <span>OCR · Translate · Prompts · Roleplay · Rewrite · Chat · Captions · Extract</span>
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

<!-- Rewrite -->
<section class="tab" id="tab-rewrite">
  <div class="col">
    <div class="card">
      <h2>Text</h2>
      <div class="modelbar" data-for="rewrite"></div>
      <div class="chips" id="rw-presets"></div>
      <div class="hint" id="rw-preset-hint" style="margin-top:6px"></div>
      <label style="margin-top:8px">Extra instruction (optional) <input id="rw-extra" placeholder="e.g. keep it under 100 words"></label>
      <label style="margin-top:8px">Text <textarea id="rw-text" class="big"></textarea></label>
      <div class="bar" style="margin-top:10px">
        <button type="button" class="primary" id="rw-run">Rewrite</button>
        <button type="button" id="rw-diff">Show changes</button>
        <button type="button" id="rw-use">Use result as source</button>
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
    <div class="krow"><span>Switch tool</span><span><kbd>Alt</kbd><kbd>1</kbd><span class="hint">–</span><kbd>8</kbd></span></div>
    <div class="krow"><span>Library</span><span><kbd>Ctrl</kbd><kbd>K</kbd></span></div>
    <div class="krow"><span>This list</span><span><kbd>Ctrl</kbd><kbd>/</kbd></span></div>
  </div>
</div>
<div id="toast" role="status"></div>

<script>
const $ = (s, r=document) => r.querySelector(s);
const $$ = (s, r=document) => [...r.querySelectorAll(s)];
const TABS = ['ocr','translate','prompt','roleplay','rewrite','chat','caption','extract'];
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
    const map = {ocr:'#ocr-run', translate:'#tr-run', prompt:'#pg-run', roleplay:'#rp-send', rewrite:'#rw-run', chat:'#ch-send', caption:'#cap-run', extract:'#ex-run'};
    const btn = $(map[currentTab()]);
    if (btn) { e.preventDefault(); btn.click(); }
  } else if (e.altKey && e.key >= '1' && e.key <= '8') {
    e.preventDefault();
    showTab(TABS[+e.key - 1]);
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

const RW_PRESETS = {
  'Formal': {instr:'Rewrite the text in a formal, professional tone.'},
  'Casual': {instr:'Rewrite the text in a relaxed, friendly, casual tone.'},
  'Shorter': {instr:'Rewrite the text so it is much shorter, and keep every key fact.'},
  'Longer': {instr:'Expand the text with smoother flow and more detail, without inventing facts.'},
  'Fix grammar': {instr:'Fix spelling, grammar and punctuation only. Keep the wording and the tone.'},
  'Humanize': {instr:'Rewrite the text so it sounds spoken by a person: varied sentence length, plain words, no stock phrases.'},
  'Screenplay': {instr:'Rewrite the text as a short screenplay beat. Action lines in the present tense. Dialogue under character names. Keep the original languages and the facts. Do not add a shot list.'},
  'Tighten dialogue': {instr:'Rewrite the dialogue so each line is shorter and more specific. Keep the speakers, the subtext and the facts. Leave action lines almost unchanged.'},
  'Subtext': {instr:'Rewrite so the feeling sits under the lines. Cut explanations of emotion. Keep the same events and the same languages.'},
  'German learner': {model:'sauerkraut-8B-Q8', hint:'Switches to sauerkraut-8B-Q8 when that model is installed.',
    instr:'I am learning German. Correct my German text. Reply in this format:\n\n**Korrigiert:** the corrected German text\n\n**Fehler:** a bullet list; for each mistake show the wrong part, then the correct part, and explain the rule briefly in English.\n\n**Tipp:** one short tip to remember.\n\nIf the text has no mistakes, say so and offer a more natural phrasing.'}
};
let RW_PRESET = 'Formal';
function pickRw(k){
  RW_PRESET = k; S('rewrite').preset = k; persist();
  drawChips('rw-presets', RW_PRESETS, k, pickRw);
  $('#rw-preset-hint').textContent = RW_PRESETS[k].hint || '';
  if (RW_PRESETS[k].model) setModel('rewrite', RW_PRESETS[k].model);
}
$('#rw-run').onclick = () => busy($('#rw-run'), async () => {
  const text = $('#rw-text').value.trim();
  if (!text) { ST.rewrite.status.textContent = 'Paste some text first.'; return; }
  const extra = $('#rw-extra').value.trim();
  const content = RW_PRESETS[RW_PRESET].instr + (extra ? '\nAlso: ' + extra : '') + '\n\nText:\n' + text;
  $('#rw-diffbox').hidden = true;
  const out = await runInto('rewrite', withSys('rewrite', [{role:'user', content}]));
  if (ST.rewrite.last !== 'err' && out.trim()) await remember('rewrite', RW_PRESET + ': ' + text.slice(0, 60), out, {model: modelOf('rewrite'), preset: RW_PRESET});
});
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
    RW_PRESET = S('rewrite').preset in RW_PRESETS ? S('rewrite').preset : 'Formal';
    drawChips('rw-presets', RW_PRESETS, RW_PRESET, pickRw);
    $('#rw-preset-hint').textContent = RW_PRESETS[RW_PRESET].hint || '';
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
    if (RW_PRESET === 'German learner') setModel('rewrite', RW_PRESETS['German learner'].model);
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
                   "tab-chat", "tab-caption", "tab-extract", "cannot read images", "Camera & blocking",
                   "Image to prompt", "Full breakdown", 'id="lib-modal"', 'id="keys-modal"',
                   "/api/library", "/api/zip", "/api/threads"):
        check("has " + marker, marker in HTML)
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
