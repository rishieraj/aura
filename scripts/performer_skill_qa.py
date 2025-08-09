#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TWO "Performer Skill Profiling" MCQs per combined video.

This script uses a sophisticated prompting strategy to handle potentially
flawed or hallucinated captions. It instructs the model to prioritize a
ground truth skill order and use visual/audio captions according to a
strict information hierarchy.
"""

import csv
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Dict, Any, List

import backoff
import openai
from tqdm import tqdm

# ─── Repository root ────────────────────────────────────────────────────────
ROOT_DIR          = Path(__file__).resolve().parent
BASE_DIR          = ROOT_DIR / "performer_skill_data"
META_CSV          = BASE_DIR / "order_log.csv"
OUTPUT_DIR        = BASE_DIR / "questions"
OUTPUT_FILE       = OUTPUT_DIR / "qa_pairs.jsonl"

# ─── OpenAI parameters ──────────────────────────────────────────────────────
MODEL_NAME        = "gpt-4o"
TEMP              = 0.6
MAX_RETRIES       = 5
PAUSE             = 1.2

CATEGORY          = "performer_skill_profiling"

# ─── System & user prompts ──────────────────────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI Benchmark Designer creating questions for the "Performer Skill Profiling" category. Your task is to generate questions that test a model's ability to differentiate between novice and expert performers by reasoning over conflicting and imperfect information sources.

You will receive separate visual and audio captions for two performances that appear one after the other in a single video. You will also receive a definitive `Ground Truth Order`.

**CRITICAL REASONING HIERARCHY:**

1.  **The `Ground Truth Order` is ABSOLUTE.** This is the undeniable fact about which performer (novice or expert) appears first. Your entire reasoning process must start from this truth.
2.  **Handle Caption Conflicts:** The provided captions may be flawed.
    * If a caption's description of skill (e.g., calling a novice an "expert") **contradicts** the `Ground Truth Order`, you MUST IGNORE the caption's faulty assessment. Instead, invent plausible, high-quality reasoning that aligns with the ground truth, using objective details from the captions (e.g., "The novice performer shows hesitant fingerwork...").
    * If the captions **align** with the ground truth, use their specific details to construct your reasoning.
3.  **Prioritize Visuals for Facts:** For identifying physical objects (instruments, clothing) and attributes (colors, locations), the `Visual Caption` is the source of truth. Use the `Audio Caption` ONLY for describing the *character* of the sound (tempo, mood, pitch, timbre, clarity, and skill cues if they align with the ground truth).

**Instructions:**
1.  **Analyze Inputs:** Read all four captions and the `Ground Truth Order` according to the hierarchy above.
2.  **Formulate TWO Distinct Questions:** Create two different questions from the following templates:
    * Template A (Identify Skill in Part): "Is the performer in the [first/second] half of the video an expert or a novice?"
    * Template B (Locate Skill Level): "Is the [expert/novice] performing in the first half or the second?"
    * Template C (Identify Attribute of Skill Level): "What is the [instrument being played by / color of the shirt worn by] the [novice/expert] player?"
3.  **Design Options:** Create a correct answer and three plausible distractors.
4.  **Write Gold Reasoning:** Justify the answer by synthesizing information **as if you were observing the video directly.** The reasoning must describe the visual and auditory evidence that supports the answer, **without mentioning the captions themselves.** For example, instead of saying "The visual caption says...", say "Visually, the performer shows...".
5.  **Output Format:** Return a **JSON list containing TWO question objects**.

---
**EXAMPLE (Handling Conflicting Captions)**

**Inputs:**
- Visual Caption (First Half): "The video captures a young woman with glasses, dressed in a gray sweater, engaging in a focused practice session with a red violin. The expertise of the performer is evident in her technique."
- Audio Caption (First Half): "A ukulele is played with a simple melody. The lack of instrumental harmonies or layered textures may indicate limited musical experience."
- Visual Caption (Second Half): "A man in a black suit plays a violin with incredible speed and passion, his posture perfect."
- Audio Caption (Second Half): "The audio is a rich, complex violin concerto played with flawless execution."
- Ground Truth Order: "The novice performs first, followed by the expert."

**Generated JSON:**
```json
[
  {
    "question": "Is the performer in the first half of the video an expert or a novice?",
    "options": {
      "A": "Expert",
      "B": "Novice",
      "C": "Intermediate",
      "D": "Impossible to determine"
    },
    "correct_answer_key": "B",
    "gold_reasoning": "The first performer is the novice. This is evident from the audio, which has a simple melody and lacks complex harmonies, suggesting limited musical experience. This contrasts with the flawless and rich execution of the expert in the second half."
  },
  {
    "question": "What instrument is the novice performer playing?",
    "options": {
      "A": "Ukulele",
      "B": "Flute",
      "C": "Piano",
      "D": "Violin"
    },
    "correct_answer_key": "D",
    "gold_reasoning": "The novice is the first performer. Visually, this performer is playing a red violin. The simple, unlayered quality of their music in the audio further supports their novice status."
  }
]
```
---
"""

USER_PROMPT_TMPL = """
Here is the data for a new video clip.

--- First Performer ---
Visual Caption:
\"\"\"{vis_first}\"\"\"
Audio Caption:
\"\"\"{aud_first}\"\"\"

--- Second Performer ---
Visual Caption:
\"\"\"{vis_second}\"\"\"
Audio Caption:
\"\"\"{aud_second}\"\"\"

Ground Truth Order:
{order}

Generate TWO distinct MCQs that satisfy all rules for the "Performer Skill Profiling" category.
"""

# ─── OpenAI client ──────────────────────────────────────────────────────────
try:
    client = openai.Client()
except openai.OpenAIError:
    print("❌  Set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# ─── Helper utilities ───────────────────────────────────────────────────────
def read_text(path: Path) -> str:
    try:
        return path.read_text(encoding="utf-8").strip()
    except Exception as e:
        tqdm.write(f"⚠️  Could not read {path}: {e}")
        return ""

def clean_json(txt: str) -> str:
    """Remove triple‑backticks and fix trailing commas so json.loads works."""
    txt = re.sub(r"```(json)?|```", "", txt, flags=re.I).strip()
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt_call(system: str, user: str) -> str:
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMP,
        messages=[{"role": "system", "content": system},
                  {"role": "user",   "content": user}],
        # response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content

def validate(item: Dict[str, Any], vid_id: str):
    required = {"question", "options", "correct_answer_key", "gold_reasoning"}
    if not required.issubset(item):
        raise ValueError(f"Missing keys {required - set(item)}")
    item["video_id"] = vid_id
    item["category"] = CATEGORY

# ─── Main workflow ──────────────────────────────────────────────────────────
def main():
    if not META_CSV.exists():
        print(f"❌  Metadata CSV not found: {META_CSV}")
        return

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Prevent duplicates if rerun
    finished_ids = set()
    if OUTPUT_FILE.exists():
        for ln in OUTPUT_FILE.open(encoding="utf-8"):
            try:
                finished_ids.add(json.loads(ln)["video_id"])
            except Exception:
                pass

    # Read CSV
    with META_CSV.open(newline="", encoding="utf-8") as csv_f:
        reader = csv.DictReader(csv_f)
        rows: List[Dict[str, str]] = list(reader)

    # Process each row
    with OUTPUT_FILE.open("a", encoding="utf-8") as out_f:
        for row in tqdm(rows, desc="Generating QA pairs", unit="video"):
            vid_stem = Path(row["combined_file"]).stem
            if vid_stem in finished_ids:
                continue

            # Resolve caption paths against ROOT_DIR
            def rel_path(key: str) -> Path:
                return ROOT_DIR / row[key]

            vis_first   = read_text(rel_path("first_visual_caption"))
            vis_second  = read_text(rel_path("second_visual_caption"))
            aud_first   = read_text(rel_path("first_audio_caption"))
            aud_second  = read_text(rel_path("second_audio_caption"))

            if not all([vis_first, vis_second, aud_first, aud_second]):
                tqdm.write(f"⚠️  Missing one or more caption files for {vid_stem}; skipping.")
                continue

            order_str = f"The {row['first_role']} performs first, followed by the {row['second_role']}."

            # Pass captions independently to the prompt template
            user_prompt = USER_PROMPT_TMPL.format(
                vis_first=vis_first,
                aud_first=aud_first,
                vis_second=vis_second,
                aud_second=aud_second,
                order=order_str
            )

            # Call OpenAI & parse
            try:
                raw_resp = gpt_call(SYSTEM_PROMPT, user_prompt)
                items = json.loads(clean_json(raw_resp))
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError("Expected a list of two QA objects.")
            except Exception as e:
                tqdm.write(f"❌  API/parse error for {vid_stem}: {e}")
                continue

            # Validate & write
            try:
                for itm in items:
                    validate(itm, vid_stem)
                    out_f.write(json.dumps(itm, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write(f"✅  Wrote 2 QAs for {vid_stem}")
            except Exception as e:
                tqdm.write(f"❌  Validation error for {vid_stem}: {e}")

            time.sleep(PAUSE)

    print(f"\n🎉  Finished. QAs saved to {OUTPUT_FILE.resolve()}")

# ─── Entry point ────────────────────────────────────────────────────────────
if __name__ == "__main__":
    main()
