import os
import json
import random
import time
import argparse
from pathlib import Path
from typing import Optional, Dict, Any, List
from tqdm import tqdm
import openai
import backoff

DEFAULT_CONFIG = {
    "data_dir": "tempo_sync_data",
    "output_dir": "questions",
    "model": "gpt-4o",
    "temperature": 0.7,
    "sleep_between": 0.2,
    "timeout": 120
}

QUESTION_CATEGORY = "tempo_av_sync_analysis"
MAX_RETRIES = 3

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
        tqdm.write(f"Shuffle error: {e}; leaving options unchanged.")
    return qa

def read_text(fp: Path) -> str:
    """Read a plain-text file; return stripped string or ''."""
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt4o_request(client: openai.Client, model: str, temp: float, timeout: int, 
                  visual: str, audio: str, sync_status: str) -> Optional[Dict[str, Any]]:
    """Call GPT-4o with retries; return parsed JSON or None."""
    content = USER_PROMPT_TEMPLATE.format(visual=visual, audio=audio, sync_status=sync_status)
    resp = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": content},
        ],
        temperature=temp,
        max_tokens=800,
        response_format={"type": "json_object"},
        timeout=timeout,
    )
    return json.loads(resp.choices[0].message.content)

def get_processed_ids(output_path: Path, resume: bool) -> set:
    """Get IDs that have already been processed."""
    done_ids = set()
    if output_path.exists() and resume:
        with output_path.open(encoding="utf-8") as f:
            for line in f:
                try:
                    done_ids.add(json.loads(line)["video_id"])
                except (json.JSONDecodeError, KeyError):
                    pass
    return done_ids

def main() -> None:
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate Tempo/AV Synchronization Analysis QA pairs using GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_CONFIG["data_dir"],
        help=f"Path to data directory (default: {DEFAULT_CONFIG['data_dir']})"
    )
    
    parser.add_argument(
        "--output-dir",
        type=str,
        default=DEFAULT_CONFIG["output_dir"],
        help=f"Path to output directory (default: {DEFAULT_CONFIG['output_dir']})"
    )
    
    parser.add_argument(
        "--api-key",
        type=str,
        help="OpenAI API key (can also use OPENAI_API_KEY env variable)"
    )
    
    parser.add_argument(
        "--model",
        type=str,
        default=DEFAULT_CONFIG["model"],
        help=f"GPT model to use (default: {DEFAULT_CONFIG['model']})"
    )
    
    parser.add_argument(
        "--temperature",
        type=float,
        default=DEFAULT_CONFIG["temperature"],
        help=f"Temperature for GPT (default: {DEFAULT_CONFIG['temperature']})"
    )
    
    parser.add_argument(
        "--sleep-between",
        type=float,
        default=DEFAULT_CONFIG["sleep_between"],
        help=f"Seconds to sleep between API calls (default: {DEFAULT_CONFIG['sleep_between']})"
    )
    
    parser.add_argument(
        "--timeout",
        type=int,
        default=DEFAULT_CONFIG["timeout"],
        help=f"Request timeout in seconds (default: {DEFAULT_CONFIG['timeout']})"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, don't resume from previous run"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    base_input_dir = Path(args.data_dir)
    aligned_vis_cap_dir = base_input_dir / "aligned_clips" / "visual_captions"
    aligned_aud_cap_dir = base_input_dir / "aligned_clips" / "audio_captions"
    misaligned_vis_cap_dir = base_input_dir / "misaligned_clips" / "visual_captions"
    misaligned_aud_cap_dir = base_input_dir / "misaligned_clips" / "audio_captions"
    output_dir = Path(args.output_dir)
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY environment variable not set or use --api-key")
        return
    
    client = openai.Client(api_key=api_key)

    output_dir.mkdir(parents=True, exist_ok=True)
    print(f"Output will be saved to: {output_dir.resolve()}")

    # --- 1. Gather all files to process ---
    files_to_process: List[Dict[str, Any]] = []
    
    # Gather aligned files
    if aligned_vis_cap_dir.is_dir():
        for vis_fp in aligned_vis_cap_dir.glob("*.txt"):
            aud_fp = aligned_aud_cap_dir / vis_fp.name
            if aud_fp.exists():
                files_to_process.append({"vis_path": vis_fp, "aud_path": aud_fp, "is_aligned": True})

    # Gather misaligned files
    if misaligned_vis_cap_dir.is_dir():
        for vis_fp in misaligned_vis_cap_dir.glob("*.txt"):
            aud_fp = misaligned_aud_cap_dir / vis_fp.name
            if aud_fp.exists():
                files_to_process.append({"vis_path": vis_fp, "aud_path": aud_fp, "is_aligned": False})
            
    if not files_to_process:
        print("Error: No caption file pairs found. Check your directory structure.")
        return

    random.shuffle(files_to_process) # Shuffle to mix aligned/misaligned
    
    # Get processed IDs if resuming
    output_path = output_dir / "qa_pairs.jsonl"
    done_ids = get_processed_ids(output_path, args.resume)
    if done_ids:
        print(f"Found {len(done_ids)} already processed IDs. Skipping them.")
        files_to_process = [f for f in files_to_process if f["vis_path"].stem not in done_ids]

    with output_path.open("a", encoding="utf-8") as f:
        for item in tqdm(files_to_process, desc="Generating QA pairs", unit="clip"):
            vis_fp = item["vis_path"]
            aud_fp = item["aud_path"]
            sync_status = "Aligned" if item["is_aligned"] else "Misaligned"
            video_id = vis_fp.stem
            
            visual_caption = read_text(vis_fp)
            audio_caption = read_text(aud_fp)
            if not (visual_caption and audio_caption):
                tqdm.write(f"Empty caption for {video_id}; skipping.")
                continue
            
            try:
                qa = gpt4o_request(client, args.model, args.temperature, args.timeout,
                                 visual_caption, audio_caption, sync_status)
                if qa:
                    qa = shuffle_qa_options(qa)
                    qa.update({"video_id": video_id, "category": QUESTION_CATEGORY, "sync_status": sync_status})
                    f.write(json.dumps(qa, ensure_ascii=False) + "\n")
                    f.flush()
                    tqdm.write(f"QA for {video_id} (Status: {sync_status})")
                else:
                    tqdm.write(f"Failed on {video_id} after retries.")
            except Exception as e:
                 tqdm.write(f"Failed on {video_id} with error: {e}")

            time.sleep(args.sleep_between)

    print(f"\nFinished. QA pairs were saved or appended to {output_path.resolve()}")

if __name__ == "__main__":
    main()
