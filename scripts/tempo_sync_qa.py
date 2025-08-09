#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Generate "Tempo/AV Synchronization Analysis" QA pairs with varied questions.

This script processes clips from both 'aligned_clips' and 'misaligned_clips'
and uses a sophisticated prompt to generate diverse, context-aware questions
that test a model's ability to detect audio-visual synchronization.
"""

# ─── Imports ────────────────────────────────────────────────────────────────
import os
import json
import random
import time
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import openai
import backoff

# ─── Config ────────────────────────────────────────────────────────────────
# --- Input Directories ---
BASE_INPUT_DIR = Path("tempo_sync_data")

ALIGNED_VIS_CAP_DIR = BASE_INPUT_DIR / "aligned_clips" / "visual_captions"
ALIGNED_AUD_CAP_DIR = BASE_INPUT_DIR / "aligned_clips" / "audio_captions"

MISALIGNED_VIS_CAP_DIR = BASE_INPUT_DIR / "misaligned_clips" / "visual_captions"
MISALIGNED_AUD_CAP_DIR = BASE_INPUT_DIR / "misaligned_clips" / "audio_captions"

# --- Output Configuration ---
QUESTION_CATEGORY = "tempo_av_sync_analysis"
OUTPUT_DIR = BASE_INPUT_DIR / "questions"

# --- GPT-4o API Settings ---
MODEL_NAME = "gpt-4o"
REQUEST_TIMEOUT = 120
MAX_RETRIES = 3
SLEEP_BETWEEN_JOBS = 0.2  # seconds

# ─── Meticulous Prompt for Varied Synchronization Analysis ──────────────────
SYSTEM_PROMPT = """
You are an expert AI Benchmark Designer creating varied and context-aware questions for the "Tempo/AV Synchronization Analysis" category. Your task is to generate a multiple-choice question that tests a model's ability to determine if a video's audio track is synchronized with its visual elements, using specific details from the scene.

**Core Task:** You will be given visual and audio descriptions and a "Ground Truth Sync Status" ('Aligned' or 'Misaligned'). You must generate a unique question that is grounded in the details provided. **DO NOT ask the same generic question every time.**

**Question Generation Strategy:**
1.  **Analyze All Inputs:** Identify the key performers, instruments, or actions in the `Visual Captions` and `Audio Captions`.
2.  **Choose a Question Template:** Randomly select one of the following templates and adapt it using the details from the captions.
    * **Template A (Specific Subject):** "How would you describe the audio-visual synchronization of the [specific performer/instrument, e.g., 'woman in the yellow dress']?"
    * **Template B (Direct Question):** "Are the actions of the [specific performer/instrument, e.g., 'violinists in black'] correctly aligned with the music being played?"
    * **Template C (Overall Assessment):** "Considering all the performers, what is the overall state of audio-visual synchronization in this clip?"
3.  **Generate Contextual Options:**
    * Create a correct answer based on the `Ground Truth Sync Status`.
    * Create plausible incorrect options. One option should be the direct opposite of the correct answer.
4.  **Write Gold Reasoning:** Justify the answer by synthesizing information **as if you were observing the video directly.** The reasoning must describe the visual and auditory evidence that supports the answer, **without mentioning the captions or the ground truth status explicitly.** For example, instead of saying "This clip is from the misaligned set," say "The performer's actions are out of sync with the audio because..."
5.  **Output Format:** Return **one** JSON object and nothing else.

---
**EXAMPLE 1**

**Inputs:**
- Visual Captions: The video captures a string quartet performance in a richly decorated hall. The musicians, dressed in formal black attire, are seated in a semi-circle.
- Audio Captions: The sound of a classical string quartet is heard. The music is complex and layered.
- Ground Truth Sync Status: Misaligned

**Generated JSON:**
{
  "question": "Regarding the string quartet in the decorated hall, are their bowing and finger movements synchronized with the audio?",
  "options": {
    "A": "Yes, their movements are perfectly in time with the music.",
    "B": "No, their movements are noticeably out of sync with the audio.",
    "C": "The cello is in sync, but the violins are not.",
    "D": "The synchronization is inconsistent, varying throughout the performance."
  },
  "correct_answer_key": "B",
  "gold_reasoning": "The bowing and fingering movements of the string quartet are visibly out of sync with the music. There is a clear temporal mismatch between the actions seen and the notes heard, indicating a desynchronized clip."
}
---
**EXAMPLE 2**

**Inputs:**
- Visual Captions: In the center, a woman in a yellow dress stands out. She is accompanied by a man in a blue shirt, who is seated and playing a guitar.
- Audio Captions: The clear sound of an acoustic guitar accompanies a female vocalist.
- Ground Truth Sync Status: Aligned

**Generated JSON:**
{
  "question": "How would you describe the audio-visual synchronization of the guitarist in the blue shirt?",
  "options": {
    "A": "His finger movements on the guitar are perfectly aligned with the sound.",
    "B": "His strumming is clearly delayed from the sound being heard.",
    "C": "He appears to be playing a different song than what is heard.",
    "D": "The woman in yellow is in sync, but the guitarist is not."
  },
  "correct_answer_key": "A",
  "gold_reasoning": "The guitarist's strumming and fretting actions are precisely timed with the guitar audio. The visual movements match the auditory rhythm and melody, confirming the performance is synchronized."
}
---
""".strip()

USER_PROMPT_TEMPLATE = """
Here is the data for a new video clip. Generate one varied and context-aware question based on the rules and examples provided.

Visual Captions:
{visual}

Audio Captions:
{audio}

Ground Truth Sync Status:
{sync_status}

Generate one MCQ that satisfies all rules for the "Tempo/AV Synchronization Analysis" category.
""".strip()

# ─── Helpers ────────────────────────────────────────────────────────────────
def shuffle_qa_options(qa: Dict[str, Any]) -> Dict[str, Any]:
    """Shuffle answer options so the correct key isn't always the same."""
    try:
        correct_key = qa["correct_answer_key"]
        correct_text = qa["options"][correct_key]
        items = list(qa["options"].items())
        random.shuffle(items)
        new_opts, new_key = {}, None
        for idx, (_, text) in enumerate(items):
            label = "ABCD"[idx]
            new_opts[label] = text
            if text == correct_text:
                new_key = label
        if new_key is None:
            raise RuntimeError("Correct answer lost during shuffle")
        qa["options"] = new_opts
        qa["correct_answer_key"] = new_key
    except Exception as e:
        tqdm.write(f"⚠️  Shuffle error: {e}; leaving options unchanged.")
    return qa

def read_text(fp: Path) -> str:
    """Read a plain-text file; return stripped string or ''."""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt4o_request(client: openai.Client, visual: str, audio: str, sync_status: str) -> Optional[Dict[str, Any]]:
    """Call GPT-4o with retries; return parsed JSON or None."""
    content = USER_PROMPT_TEMPLATE.format(visual=visual, audio=audio, sync_status=sync_status)
    resp = client.chat.completions.create(
        model=MODEL_NAME,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=0.7, # Increased temperature for more variety
        max_tokens=800,
        response_format={"type": "json_object"},
        timeout=REQUEST_TIMEOUT,
    )
    return json.loads(resp.choices[0].message.content)

# ─── Main ────────────────────────────────────────────────────────────────────
def main() -> None:
    if not os.getenv("OPENAI_API_KEY"):
        print("❌  OPENAI_API_KEY environment variable not set.")
        return
    client = openai.Client()

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    print(f"📂 Output will be saved to: {OUTPUT_DIR.resolve()}")

    # --- 1. Gather all files to process ---
    files_to_process: List[Dict[str, Any]] = []
    
    # Gather aligned files
    if ALIGNED_VIS_CAP_DIR.is_dir():
        for vis_fp in ALIGNED_VIS_CAP_DIR.glob("*.txt"):
            aud_fp = ALIGNED_AUD_CAP_DIR / vis_fp.name
            if aud_fp.exists():
                files_to_process.append({"vis_path": vis_fp, "aud_path": aud_fp, "is_aligned": True})

    # Gather misaligned files
    if MISALIGNED_VIS_CAP_DIR.is_dir():
        for vis_fp in MISALIGNED_VIS_CAP_DIR.glob("*.txt"):
            aud_fp = MISALIGNED_AUD_CAP_DIR / vis_fp.name
            if aud_fp.exists():
                files_to_process.append({"vis_path": vis_fp, "aud_path": aud_fp, "is_aligned": False})
            
    if not files_to_process:
        print("❌ Error: No caption file pairs found. Check your directory structure.")
        print(f"   - Looked in: {ALIGNED_VIS_CAP_DIR} & {ALIGNED_AUD_CAP_DIR}")
        print(f"   - Looked in: {MISALIGNED_VIS_CAP_DIR} & {MISALIGNED_AUD_CAP_DIR}")
        return

    random.shuffle(files_to_process) # Shuffle to mix aligned/misaligned

    # --- 2. Generate QA pairs ---
    output_path = OUTPUT_DIR / "qa_pairs.jsonl"
    with output_path.open("a", encoding="utf-8") as f:
        for item in tqdm(files_to_process, desc="Generating QA pairs", unit="clip"):
            vis_fp = item["vis_path"]
            aud_fp = item["aud_path"]
            sync_status = "Aligned" if item["is_aligned"] else "Misaligned"
            video_id = vis_fp.stem
            
            visual_caption = read_text(vis_fp)
            audio_caption = read_text(aud_fp)
            if not (visual_caption and audio_caption):
                tqdm.write(f"⚠️  Empty caption for {video_id}; skipping.")
                continue
            
            try:
                qa = gpt4o_request(client, visual_caption, audio_caption, sync_status)
                if qa:
                    qa = shuffle_qa_options(qa)
                    qa.update({"video_id": video_id, "category": QUESTION_CATEGORY, "sync_status": sync_status})
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    f.flush()
                    tqdm.write(f"✅  QA for {video_id} (Status: {sync_status})")
                else:
                    tqdm.write(f"❌  Failed on {video_id} after retries.")
            except Exception as e:
                 tqdm.write(f"❌  Failed on {video_id} with error: {e}")

            time.sleep(SLEEP_BETWEEN_JOBS)

    print(f"\n🎉  Finished. QA pairs were saved or appended to {output_path.resolve()}")

if __name__ == "__main__":
    main()
