import os, sys, re
import json
import time
import argparse
from pathlib import Path
from typing import List, Dict, Any
from tqdm import tqdm
import openai
import backoff

DEFAULT_CONFIG = {
    "data_dir": "causal_reasoning_data",
    "output_dir": "questions",
    "model": "gpt-4o",
    "temperature": 0.5,
    "sleep_between": 1.0
}

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

def normalise_json_str(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```(json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=5)
def gpt_call(client, model: str, temp: float, system_prompt: str, user_prompt: str) -> str:
    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ]
    )
    return resp.choices[0].message.content

def validate_item(d: Dict[str, Any], vid: str):
    required = {"question", "options", "correct_answer_key", "gold_reasoning", "video_id", "category"}
    if not required.issubset(d):
        raise ValueError(f"Missing keys: {required - set(d.keys())}")
    d["video_id"] = vid
    d["category"] = "causal_reasoning"

def read_text(fp: Path) -> str:
    try:
        return fp.read_text(encoding="utf-8").strip()
    except Exception:
        return ""

def get_processed_ids(output_path: Path, resume: bool) -> set:
    done_ids = set()
    if output_path.exists() and resume:
        with output_path.open("r", encoding="utf-8") as f:
            for ln in f:
                try:
                    done_ids.add(json.loads(ln)["video_id"])
                except:
                    pass
    return done_ids

def main():
    parser = argparse.ArgumentParser(
        description="Generate Cross-Modal Causal Reasoning MCQs using GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--data-dir", type=str, default=DEFAULT_CONFIG["data_dir"],
                       help=f"Path to data directory (default: {DEFAULT_CONFIG['data_dir']})")
    parser.add_argument("--output-dir", type=str, default=DEFAULT_CONFIG["output_dir"],
                       help=f"Path to output directory (default: {DEFAULT_CONFIG['output_dir']})")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--model", type=str, default=DEFAULT_CONFIG["model"],
                       help=f"GPT model to use (default: {DEFAULT_CONFIG['model']})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"],
                       help=f"Temperature (default: {DEFAULT_CONFIG['temperature']})")
    parser.add_argument("--sleep-between", type=float, default=DEFAULT_CONFIG["sleep_between"],
                       help=f"Sleep between calls (default: {DEFAULT_CONFIG['sleep_between']})")
    parser.add_argument("--no-resume", action="store_false", dest="resume",
                       help="Start fresh, ignore previous progress")
    
    args = parser.parse_args()
    
    # Setup paths
    base_data_dir = Path(args.data_dir)
    transcripts_dir = base_data_dir / "transcripts"
    vis_captions_dir = base_data_dir / "visual_captions"
    aud_captions_dir = base_data_dir / "audio_captions"
    output_dir = Path(args.output_dir)
    
    # Check directories
    if not base_data_dir.exists():
        print(f"Error: Data directory '{base_data_dir}' not found")
        sys.exit(1)
    
    for dir_path in [transcripts_dir, vis_captions_dir, aud_captions_dir]:
        if not dir_path.exists():
            print(f"Error: Missing subdirectory {dir_path}")
            sys.exit(1)
    
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Initialize OpenAI
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: OpenAI API key not found. Set OPENAI_API_KEY or use --api-key")
        sys.exit(1)
    
    try:
        client = openai.Client(api_key=api_key)
    except Exception as e:
        print(f"Failed to initialize OpenAI client: {e}")
        sys.exit(1)
    
    # Get transcript files
    all_transcript_files = list(transcripts_dir.glob("*.txt"))
    if not all_transcript_files:
        print(f"No transcript files found in {transcripts_dir}")
        return
    
    print(f"Found {len(all_transcript_files)} clips to process")
    
    # Setup output
    output_path = output_dir / "qa_pairs.jsonl"
    done_ids = get_processed_ids(output_path, args.resume)
    
    if done_ids:
        print(f"Found {len(done_ids)} already processed clips. Resuming...")
    
    # Filter files
    files_to_process = [fp for fp in all_transcript_files if fp.stem not in done_ids]
    
    if not files_to_process:
        print("All clips have been processed!")
        return
    
    # Process clips
    success_count = 0
    error_count = 0
    
    with output_path.open("a", encoding="utf-8") as out_f:
        for trn_fp in tqdm(files_to_process, desc="Generating QA pairs", unit="clip"):
            vid = trn_fp.stem
            
            vis_fp = vis_captions_dir / f"{vid}.txt"
            aud_fp = aud_captions_dir / f"{vid}.txt"
            
            if not (vis_fp.exists() and aud_fp.exists()):
                tqdm.write(f"Missing caption files for {vid}; skipping")
                error_count += 1
                continue
            
            transcript_text = read_text(trn_fp)
            visual_caption = read_text(vis_fp)
            audio_caption = read_text(aud_fp)
            
            if not (visual_caption and audio_caption):
                tqdm.write(f"Empty captions for {vid}; skipping")
                error_count += 1
                continue
            
            user_prompt = USER_PROMPT_TMPL.format(
                vid=vid, visual=visual_caption, audio=audio_caption, transcript=transcript_text
            )
            
            try:
                raw = gpt_call(client, args.model, args.temperature, SYSTEM_PROMPT, user_prompt)
                items = json.loads(normalise_json_str(raw))
                
                if not isinstance(items, list) or len(items) != 2:
                    raise ValueError(f"Expected 2 items, got {len(items)}")
                
                for qa_item in items:
                    validate_item(qa_item, vid)
                    out_f.write(json.dumps(qa_item, ensure_ascii=False) + "\n")
                out_f.flush()
                
                success_count += 1
                tqdm.write(f"Generated 2 QAs for {vid}")
                
            except Exception as e:
                error_count += 1
                tqdm.write(f"ERROR for {vid}: {e}")
            
            time.sleep(args.sleep_between)
    
    print(f"\nProcessing complete!")
    print(f"Successfully processed: {success_count} clips")
    print(f"Failed: {error_count} clips")
    print(f"Output saved to: {output_path}")

if __name__ == "__main__":
    main()
