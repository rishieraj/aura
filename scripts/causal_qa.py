#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TWO "Cross-Modal Causal Reasoning" MCQs per clip using GPT-4o.

This script reads data from a consolidated 'causal_reasoning_data' folder,
using visual captions, audio captions, and transcripts as input. It uses an
advanced prompt to generate high-quality, caption-agnostic reasoning.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os, sys, re
import json
import random
import time
from pathlib import Path
from typing import Optional, List, Dict, Any
from tqdm import tqdm
import openai
import backoff

# ─── Config ────────────────────────────────────────────────────────────────
# --- Input Directories ---
BASE_DATA_DIR = Path("causal_reasoning_data")
TRANSCRIPTS_DIR = BASE_DATA_DIR / "transcripts"
VIS_CAPTIONS_DIR = BASE_DATA_DIR / "visual_captions"
AUD_CAPTIONS_DIR = BASE_DATA_DIR / "audio_captions"

# --- Output Configuration ---
OUTPUT_DIR = Path("questions")

# --- GPT-4o API Settings ---
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.5
MAX_RETRIES = 5
SLEEP_BETWEEN = 1.0

# ─── OpenAI Client Initialization ───────────────────────────────────────────
try:
    client = openai.Client()
except openai.OpenAIError:
    print("❌  Please set OPENAI_API_KEY in your environment.")
    sys.exit(1)

# ─── Meticulous Prompt for Causal Reasoning ──────────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI Benchmark Designer creating questions for the "Cross-Modal Causal Reasoning" category. Your task is to generate questions that probe a model's ability to connect a cause in one modality with an effect in another.

**Core Rule: The question MUST be unanswerable from a single modality.** A user must synthesize information from both the visual and audio/transcript descriptions to deduce the correct answer. The question should specifically ask for the *cause* of a visual event (which is explained in the audio/transcript) or the *effect* of an audio event (which is seen in the visuals).

**Information Hierarchy:**
* **Trust `Visual Captions` for identifying physical objects and their attributes.**
* **Use `Audio Captions` and `Transcripts` for auditory character and spoken information.** The transcript is often the source of the "reason" or "cause."
* **Ignore Nonsensical Transcripts:** If the `Whisper Transcript` contains gibberish, filler words (e.g., 'uhm', 'ah'), or is clearly unrelated to the scene, it should be disregarded as a source of causal information.

**Instructions:**
1.  **Analyze All Inputs:** Carefully read the Visual Captions, Audio-only Captions, and the Whisper Transcript.
2.  **Identify TWO Causal Links:** Find two clear cause-and-effect relationships where the cause and effect are in different modalities.
3.  **Formulate TWO Distinct Questions:** Create two different questions that explicitly ask "Why is [effect] happening?" or "What is the result of [cause]?".
4.  **Design Trap Options:** For each question, create a correct answer and three plausible distractors (e.g., a visual-only trap, an audio-only trap).
5.  **Write Gold Reasoning:** Justify the answer by synthesizing information **as if you were observing the video directly.** The reasoning must describe the visual and auditory evidence that supports the answer, **without mentioning the captions or transcripts themselves.**
6.  **Output Format:** Return a **JSON list containing TWO question objects**.

---
**EXAMPLE**

**Inputs:**
- Visual Captions: "A person is carefully attaching a white panel to a bright yellow canopy structure using a series of black clamps."
- Audio-only Captions: "a man is speaking with background noise"
- Whisper Transcript: "After this, you will need to connect the top velcro to the frame. This ensures that the top stays taught and all water runs off easily."

**Generated JSON:**
```json
[
  {
    "question": "What is the underlying reason for securing the panel to the yellow canopy?",
    "options": {
      "A": "To add a final decorative touch to the canopy.",
      "B": "To make the canopy top taut and ensure water runs off.",
      "C": "To perform a necessary repair on a broken frame section.",
      "D": "To demonstrate how to use clamps for a general purpose."
    },
    "correct_answer_key": "B",
    "gold_reasoning": "The visual action of attaching the panel is directly explained by the speaker's instructions. The spoken words clarify that the purpose is to make the canopy taut and ensure water runs off.",
    "video_id": "placeholder_id",
    "category": "causal_reasoning"
  },
  {
    "question": "What is the direct result of the speaker's instruction to 'connect the top velcro'?",
    "options": {
        "A": "The person in the video begins sewing a new panel.",
        "B": "The person is seen attaching a white panel to the frame.",
        "C": "The canopy collapses due to incorrect assembly.",
        "D": "The speaker stops talking and music begins to play."
    },
    "correct_answer_key": "B",
    "gold_reasoning": "The spoken instruction to 'connect the top velcro' is the direct cause of the visual event. Immediately following this instruction, the person is seen physically attaching the white panel to the yellow frame.",
    "video_id": "placeholder_id",
    "category": "causal_reasoning"
  }
]
```
---
"""

USER_PROMPT_TMPL = """\
Video ID: {vid}

Visual Captions:
{visual}

Audio-only Captions:
{audio}

Whisper Transcript:
{transcript}

Generate TWO distinct MCQs that satisfy all rules.
"""
# ----------------------------------------------------------------------

def normalise_json_str(txt: str) -> str:
    """Fix common mistakes like trailing commas or ``` blocks."""
    txt = txt.strip()
    txt = re.sub(r"^```(json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt_call(system_prompt: str, user_prompt: str) -> str:
    """Makes an API call to OpenAI using the modern client syntax."""
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        temperature=TEMPERATURE,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user",   "content": user_prompt}
        ],
        # response_format={"type": "json_object"}
    )
    return resp.choices[0].message.content

def validate_item(d: Dict[str, Any], vid: str):
    required = {"question", "options", "correct_answer_key",
                "gold_reasoning", "video_id", "category"}
    if not required.issubset(d):
        raise ValueError(f"Missing keys: {required - set(d.keys())}")
    d["video_id"] = vid
    d["category"] = "causal_reasoning"

def read_text(fp: Path) -> str:
    """Read a plain-text file; return stripped string or ''."""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception as e:
        tqdm.write(f"⚠️  Could not read {fp}: {e}")
        return ""

def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    
    if not TRANSCRIPTS_DIR.is_dir():
        print(f"❌ Error: Transcript directory not found at '{TRANSCRIPTS_DIR}'")
        return
        
    all_transcript_files = list(TRANSCRIPTS_DIR.glob("*.txt"))
    
    if not all_transcript_files:
        print(f"❌ No transcript files found in '{TRANSCRIPTS_DIR}'.")
        return

    print(f"🔍 Found {len(all_transcript_files)} potential clips to process.")

    output_path = OUTPUT_DIR / "qa_pairs.jsonl"
    done_ids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    done_ids.add(json.loads(ln)["video_id"])
                except Exception:
                    pass
    
    if done_ids:
        print(f"Found {len(done_ids)} previously generated IDs. Skipping them.")

    with output_path.open("a", encoding="utf-8") as out_f:
        files_to_process = [fp for fp in all_transcript_files if fp.stem not in done_ids]
        for trn_fp in tqdm(files_to_process, desc="Generating QA pairs", unit="clip"):
            vid = trn_fp.stem
            
            vis_fp = VIS_CAPTIONS_DIR / f"{vid}.txt"
            aud_fp = AUD_CAPTIONS_DIR / f"{vid}.txt"

            if not (vis_fp.exists() and aud_fp.exists()):
                tqdm.write(f"⚠️  Missing a caption file for {vid}; skipping.")
                continue

            transcript_text = read_text(trn_fp)
            visual_caption  = read_text(vis_fp)
            audio_caption   = read_text(aud_fp)

            if not (visual_caption and audio_caption):
                tqdm.write(f"⚠️  Empty visual or audio caption for {vid}; skipping.")
                continue

            user_prompt = USER_PROMPT_TMPL.format(
                vid=vid, visual=visual_caption, audio=audio_caption, transcript=transcript_text
            )

            try:
                raw = gpt_call(SYSTEM_PROMPT, user_prompt)
                items = json.loads(normalise_json_str(raw))
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError(f"Expected a list of 2 items, but got {len(items)}.")
            except Exception as e:
                tqdm.write(f"❌  {vid}: API call, parse, or validation error: {e}")
                continue
            
            try:
                for qa_item in items:
                    validate_item(qa_item, vid)
                    out_f.write(json.dumps(qa_item, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write(f"✅  Successfully generated 2 QAs for {vid}")
            except Exception as e:
                tqdm.write(f"❌  {vid}: Validation error: {e}")
                continue

            time.sleep(SLEEP_BETWEEN)

    print(f"\n✅  All questions written to {output_path}")

if __name__ == "__main__":
    main()
