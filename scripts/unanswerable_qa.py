#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate TWO "Cross-Modal Unanswerability" MCQs per clip using GPT-4o.

This script reads data from a consolidated data folder, using visual captions, 
audio captions, and transcripts as input to generate questions that are 
impossible to answer from the provided context. The goal is to test a model's
ability to avoid hallucination and recognize information gaps.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import sys
import re
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import openai
import backoff

# ─── Config ────────────────────────────────────────────────────────────────

# --- Input Directories ---
BASE_DATA_DIR = Path("unanswerability_data")
TRANSCRIPTS_DIR = BASE_DATA_DIR / "transcripts"
VIS_CAPTIONS_DIR = BASE_DATA_DIR / "visual_captions"
AUD_CAPTIONS_DIR = BASE_DATA_DIR / "audio_captions"

# --- Output Configuration ---
OUTPUT_DIR = BASE_DATA_DIR / "questions"

# --- GPT-4o API Settings ---
MODEL_NAME = "gpt-4o"
TEMPERATURE = 0.5
MAX_RETRIES = 5
SLEEP_BETWEEN = 1.0

# ─── OpenAI Client Initialization ───────────────────────────────────────────
try:
    client = openai.Client()
except openai.OpenAIError:
    print("❌  Error: Please set the OPENAI_API_KEY environment variable.")
    sys.exit(1)

# ─── Meticulous Prompt for Unanswerability Task ─────────────────────────────
SYSTEM_PROMPT = """
You are an expert AI Benchmark Designer creating questions for the "Cross-Modal Unanswerability" category. Your task is to generate questions that are **impossible** to answer from the video and audio content.

**Core Rule: The question must probe for a detail that is plausible but ABSENT from the visual, audio, and transcript data.** The goal is to test a model's ability to recognize the limits of the available information and avoid hallucinating an answer.

**Instructions:**
1.  **Analyze All Inputs:** Carefully read the Visual Captions, Audio-only Captions, and the Whisper Transcript.
2.  **Identify Information Gaps:** Find **two distinct** details, objects, or events that are hinted at or could plausibly exist but are not actually described. For example, the context might mention a "performer," but not what they are performing, or a "meeting," but not its specific topic.
3.  **Formulate TWO Distinct Questions:** Create two different questions, each asking about one of the specific, absent pieces of information.
4.  **Design Trap and Correct Options:** For each question:
    * **Distractor Options (3):** Create three incorrect but plausible-sounding options that a model might incorrectly infer from the context.
    * **Correct Answer (1):** One option must explicitly state that the information is not available or cannot be determined from the video/audio.
5.  **Write Gold Reasoning:** Justify the answer by synthesizing information **as if you were observing the video directly.** The reasoning must explain *why* the question is unanswerable by stating that the necessary visual or auditory evidence is not present in the scene, **without mentioning the captions or transcripts themselves.**
6.  **Output Format:** Return a **JSON list containing TWO question objects**.

---
**EXAMPLE**

**Inputs:**
- Visual Captions: "A vibrant street is filled with people. The scene is enhanced by the presence of street performers."
- Audio-only Captions: "A male voice delivers a passionate speech with intermittent impact sounds."
- Whisper Transcript: "I'm not sure I'm still alive. Blocked as well. Everybody just took too long."

**Generated JSON:**
```json
[
  {
    "question": "What specific act are the street performers engaged in?",
    "options": {
      "A": "They are playing musical instruments.",
      "B": "They are performing a silent mime routine.",
      "C": "They are juggling colorful balls and clubs.",
      "D": "The specific activity of the street performers is not shown or described."
    },
    "correct_answer_key": "D",
    "gold_reasoning": "While street performers are visually present in the scene, their specific actions are not shown. The audio consists of a passionate speech, which is unrelated to their performance, making it impossible to determine their activity.",
    "video_id": "placeholder_id",
    "category": "unanswerability"
  },
  {
    "question": "What is the source of the intermittent impact sounds heard in the audio?",
    "options": {
        "A": "The sounds are from a construction site nearby.",
        "B": "The speaker is stomping their foot for emphasis.",
        "C": "The source of the impact sounds is not visually identifiable.",
        "D": "The sounds are from the street performers' act."
    },
    "correct_answer_key": "C",
    "gold_reasoning": "The audio contains intermittent impact sounds, but the video does not show the source of these noises. Without a visual anchor, it is impossible to determine what is causing them.",
    "video_id": "placeholder_id",
    "category": "unanswerability"
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

Generate TWO distinct MCQs that satisfy all rules for the unanswerability task.
"""
# ----------------------------------------------------------------------

def normalise_json_str(txt: str) -> str:
    """Cleans the raw string response from the API."""
    txt = txt.strip()
    txt = re.sub(r"^```(json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt_call(system_prompt: str, user_prompt: str) -> str:
    """Makes a robust API call to the OpenAI ChatCompletion endpoint."""
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
    """Validates the structure of a single generated JSON object."""
    required = {"question", "options", "correct_answer_key",
                "gold_reasoning", "video_id", "category"}
    if not required.issubset(d):
        raise ValueError(f"Missing keys: {required - set(d.keys())}")
    
    if len(d.get("options", {})) != 4:
        raise ValueError("The 'options' dictionary must contain exactly 4 choices.")
        
    d["video_id"] = vid
    d["category"] = "unanswerability"

def read_text(fp: Path) -> str:
    """Reads a plain-text file and returns its stripped content."""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception as e:
        tqdm.write(f"⚠️  Could not read {fp}: {e}")
        return ""

def main():
    """Main execution function."""
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
                except (json.JSONDecodeError, KeyError):
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

            user_prompt = USER_PROMPT_TMPL.format(
                vid=vid, visual=visual_caption, audio=audio_caption, transcript=transcript_text
            )

            try:
                raw_response = gpt_call(SYSTEM_PROMPT, user_prompt)
                items = json.loads(normalise_json_str(raw_response))
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError(f"Expected a list of 2 items, but got {len(items)}.")
            except Exception as e:
                tqdm.write(f"❌  ERROR for {vid} (API/Parse): {e}")
                continue

            try:
                for qa_item in items:
                    validate_item(qa_item, vid)
                    out_f.write(json.dumps(qa_item, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write(f"✅  Successfully generated 2 QAs for {vid}")
            except Exception as e:
                tqdm.write(f"❌  ERROR for {vid} (Validation): {e}")
                continue

            time.sleep(SLEEP_BETWEEN)

    print(f"\n✅  Processing complete. All questions written to {output_path}")

if __name__ == "__main__":
    main()
