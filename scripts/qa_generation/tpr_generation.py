import json
import os
import sys
import time
import re
import argparse
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import openai
import backoff

DEFAULT_CONFIG = {
    "data_dir": "pitch_timbre_data",
    "output_dir": "questions",
    "model": "gpt-4o",
    "temperature": 0.5,
    "pause": 1.2
}

CATEGORY = "pitch_timbre_reasoning"
MAX_RETRIES = 5

SYSTEM_PROMPT = """
You are an expert AI Benchmark Designer creating highly specific questions for the "Pitch/Timbre Reasoning" category. Your task is to generate questions that test a model's ability to connect a fine-grained, **comparative** auditory quality (pitch or timbre) with a precise visual detail.

**Core Task: Probing Comparative Auditory Qualities**
Your primary goal is to create questions that CANNOT be answered without first making a judgment about the relative pitch or timbre of different sound sources. The questions **MUST** contain comparative auditory terms.

* **Good Question Examples (Use this structure):**
    * "What is the color of the shirt worn by the person playing the **higher-pitched** instrument?"
    * "Does the person playing the instrument with a **heavier timbre** have longer hair?"
    * "Of the two singers, which one has a **deeper voice**?"

* **Bad Question Examples (AVOID THESE):**
    * "What color is the guitar?" (This is just visual recognition.)
    * "What instrument is being played?" (This is just audio recognition.)

**Information & Reasoning Hierarchy:**

1.  **Visuals are Truth for Facts:** For identifying physical objects (instruments, clothing) and attributes (colors, locations), the `Visual Caption` is the source of truth.
2.  **Audio is Truth for Character:** Use the `Audio Caption` for describing the *mood*, *tempo*, and *quality* of the sound. It may contain useful comparative terms (e.g., "a deep bass sound accompanies a high-pitched melody").
3.  **Use Inherent Knowledge as a Fallback:** If the captions do not explicitly compare the sounds, use your general knowledge about the instruments or voices identified in the visual captions to infer their relative pitch or timbre (e.g., a bass guitar is inherently lower-pitched than a flute; a man's voice is typically deeper than a woman's).
4.  **Write Caption-Agnostic Reasoning:** Justify the answer by synthesizing information **as if you were observing the video directly.** The reasoning must describe the visual and auditory evidence that supports the answer, **without mentioning the captions themselves.**

**Instructions:**
1.  **Analyze Inputs:** Read the Visual and Audio Captions according to the hierarchy above.
2.  **Formulate THREE Distinct Questions:** Create three different multiple-choice questions that follow the "Core Task" guidelines. Each question must probe a different pitch/timbre comparison or link it to a different visual attribute.
3.  **Design Options:** Create a correct answer and plausible distractors. One distractor should ideally be a detail associated with the *other* sound source in the comparison.
4.  **Output Format:** Return a **JSON list containing THREE question objects**.

---
**EXAMPLE**

**Inputs:**
- Visual Captions: "The video features a man and a woman seated closely together, each holding a guitar. The man, dressed in a dark blue shirt, has short hair. The woman, wearing a floral top, has long hair. A framed picture is on the wall behind them."
- Audio-only Captions: "A male vocalist sings a slow, melodic tune with a deep voice. A higher-pitched female harmony is also audible, accompanied by two acoustic guitars, one with a brighter, thinner timbre."

**Generated JSON:**
```json
[
  {
    "question": "What is the shirt color of the person singing with the deeper voice?",
    "options": { "A": "Floral", "B": "Dark blue", "C": "White", "D": "Black" },
    "correct_answer_key": "B",
    "gold_reasoning": "The audio contains a deep male voice. Visually, the male musician is wearing a dark blue shirt. The floral shirt is a distractor as it is worn by the female musician with the higher voice.",
    "video_id": "placeholder_id",
    "category": "pitch_timbre_reasoning"
  },
  {
    "question": "Does the person playing the guitar with the brighter, thinner timbre have long or short hair?",
    "options": { "A": "Long hair", "B": "Short hair", "C": "No hair", "D": "A hat" },
    "correct_answer_key": "A",
    "gold_reasoning": "The audio distinguishes one guitar as having a brighter, thinner timbre, which typically corresponds to a higher pitch. This sound would be associated with the female musician, who is visually described as having long hair.",
    "video_id": "placeholder_id",
    "category": "pitch_timbre_reasoning"
  },
  {
    "question": "Of the two singers, which one has the higher-pitched voice?",
    "options": { "A": "The person in the dark blue shirt", "B": "The person with long hair", "C": "They sing at the same pitch", "D": "There is only one singer" },
    "correct_answer_key": "B",
    "gold_reasoning": "The audio clearly features two singers, one with a deep voice and one with a higher-pitched harmony. The higher voice belongs to the female musician, who is visually identified by her long hair.",
    "video_id": "placeholder_id",
    "category": "pitch_timbre_reasoning"
  }
]
```
---
"""

USER_PROMPT_TEMPLATE = """
Here is the data for a new video clip.

Visual Captions:
{visual}

Audio-only Captions:
{audio}

Generate THREE distinct MCQs that satisfy all rules for the "Pitch/Timbre Reasoning" category, returning them in a single JSON list.
"""

def read_text(p: Path) -> str:
    """Safely read a text file, returning its content or an empty string."""
    if p and p.exists():
        try:
            return p.read_text(encoding="utf-8").strip()
        except Exception as e:
            tqdm.write(f"Could not read {p}: {e}")
    return ""

def clean_json_str(txt: str) -> str:
    """Clean the raw string from the API to make it valid JSON."""
    txt = re.sub(r"```(json)?|```", "", txt).strip()
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def gpt_call(client, model: str, temp: float, system: str, user: str) -> str:
    """Make a robust API call with backoff for rate limiting."""
    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": system},
            {"role": "user", "content": user}
        ],
    )
    return resp.choices[0].message.content

def validate_item(item: Dict[str, Any], vid: str):
    """Validate a single generated question dictionary."""
    required_keys = {"question", "options", "correct_answer_key", "gold_reasoning"}
    if not required_keys.issubset(item):
        raise ValueError(f"Item missing required keys: {required_keys - set(item.keys())}")
    item["video_id"] = vid
    item["category"] = CATEGORY

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

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate Pitch/Timbre Reasoning MCQs using GPT-4o",
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
        "--pause",
        type=float,
        default=DEFAULT_CONFIG["pause"],
        help=f"Seconds to pause between API calls (default: {DEFAULT_CONFIG['pause']})"
    )
    
    parser.add_argument(
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, don't resume from previous run"
    )
    
    args = parser.parse_args()
    
    # Setup directories
    base_data_dir = Path(args.data_dir)
    vis_captions_dir = base_data_dir / "visual_captions"
    aud_captions_dir = base_data_dir / "audio_captions"
    output_dir = Path(args.output_dir)
    
    # Initialize OpenAI client
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("Error: Please set the OPENAI_API_KEY environment variable or use --api-key")
        sys.exit(1)
    
    try:
        client = openai.Client(api_key=api_key)
    except openai.OpenAIError as e:
        print(f"Error initializing OpenAI client: {e}")
        sys.exit(1)
    
    # Main processing
    output_dir.mkdir(parents=True, exist_ok=True)
    output_path = output_dir / "qa_pairs.jsonl"

    vis_files = sorted(vis_captions_dir.glob("*.txt"))
    if not vis_files:
        print(f"No visual caption files found in {vis_captions_dir}")
        return

    done_ids = get_processed_ids(output_path, args.resume)
    if done_ids:
        print(f"Found {len(done_ids)} already processed IDs. Skipping them.")

    with output_path.open("a", encoding="utf-8") as out_f:
        files_to_process = [fp for fp in vis_files if fp.stem not in done_ids]
        
        for vis_fp in tqdm(files_to_process, desc="Generating QA pairs", unit="clip"):
            vid = vis_fp.stem

            audio_fp = aud_captions_dir / f"{vid}.txt"
            if not audio_fp.exists():
                tqdm.write(f"Missing audio caption for {vid}; skipping.")
                continue

            visual_text = read_text(vis_fp)
            audio_text = read_text(audio_fp)
            
            user_prompt = USER_PROMPT_TEMPLATE.format(
                visual=visual_text, audio=audio_text
            )

            try:
                raw_response = gpt_call(client, args.model, args.temperature, 
                                      SYSTEM_PROMPT, user_prompt)
                items = json.loads(clean_json_str(raw_response))
                if not isinstance(items, list) or len(items) != 3:
                    raise ValueError(f"Expected a list of 3 items, but got {len(items)}.")
            except Exception as e:
                tqdm.write(f"ERROR for {vid} (API/Parse): {e}")
                continue

            try:
                for item in items:
                    validate_item(item, vid)
                    out_f.write(json.dumps(item, ensure_ascii=False) + "\n")
                out_f.flush()
                tqdm.write(f"Successfully generated 3 QAs for {vid}")
            except Exception as e:
                tqdm.write(f"ERROR for {vid} (Validation): {e}")
            
            time.sleep(args.pause)

    print(f"\nProcessing complete. Questions appended to {output_path.resolve()}")

if __name__ == "__main__":
    main()
