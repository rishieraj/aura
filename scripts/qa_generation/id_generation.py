import openai, csv, re, json, sys, time, backoff, argparse
from pathlib import Path
from typing import Dict, Any
from tqdm import tqdm

DEFAULT_CONFIG = {
    "data_dir": "implicit_distractions_data",
    "output_file": "implicit_questions.jsonl",
    "model": "gpt-4o",
    "temperature": 0.4,
    "sleep_between": 1.5
}

MAX_RETRIES = 5

SYS_PROMPT = """You are an expert AI Benchmark Designer specializing in creating challenging multiple-choice questions. Your task is to test a model's ability to handle specific spatial references and avoid "implicit distractions" or attentional errors.

You will receive separate **Visual** and **Audio** captions for two different clips that have been stitched together into a single video, one in the top half and one in the bottom half. A key "target" object or performer (e.g., a guitarist) often appears in **both** halves of the video.

The goal is to craft questions that force a model to correctly ground its answer in the specified half (e.g., "the guitarist at the top") and ignore the visually and audibly similar but contextually incorrect information from the other half.

**Information Hierarchy (Very Important):**
* **Trust `Visual Captions` for identifying physical objects and their attributes.** The visual description is the ground truth for what instruments are present, what people are wearing, and where they are located.
* **Use `Audio Captions` for auditory character only.** The audio descriptions are useful for understanding the *mood*, *tempo*, and *quality* of the sound, but they may occasionally misidentify the source of the sound (e.g., call a flute a clarinet). **If the visual and audio captions disagree on an object, you must prioritize the visual caption.**

**CRITICAL INSTRUCTIONS:**

1.  **Identify the Target:** First, read the **Visual and Audio captions for both halves** and identify the common object, instrument, or performer that appears in both. This is your "target".
2.  **Find Distinguishing Details:** For this target, find unique, fine-grained **visual or auditory details** specific to **each** half. Examples: the color of their shirt, the instrument next to them (visual), or the tempo of the music, a specific sound effect (audio).
3.  **Craft TWO Distinct Questions:** For each set of captions, you must generate **two** separate multiple-choice questions.
    * One question must be about the target in one half (e.g., the top).
    * The second question must be about the target in the other half (e.g., the bottom).
4.  **Design "Hallucination Trap" Options:** This is the most important rule. The incorrect answer options MUST be designed as traps. At least one incorrect option should be a detail that is true for the target in the *other* half of the video.
5.  **Write Precise Gold Reasoning:** The `gold_reasoning` is essential. It must explicitly state:
    * Which half of the video (e.g., "top", "bottom") contains the correct evidence.
    * The specific visual or auditory detail that justifies the correct answer.
    * Mention the detail from the other half that serves as the hallucination trap.
6.  **Strict JSON Output:** Return **only** a valid JSON list containing exactly two dictionary objects. Do not include any other text or markdown formatting.

---
**EXAMPLE 1**

**Caption:** "The video is a vibrant collage. In the top half, a man with curly hair and a beard plays an acoustic guitar next to a woman on a red couch. The bottom half shows a young man with short hair playing an electric guitar in front of a brick wall."

**Generated JSON:**
```json
[
  {
    "question": "What is the background behind the guitarist in the top half of the video?",
    "options": {
      "A": "A red couch",
      "B": "A brick wall",
      "C": "A window with a view of the city",
      "D": "A large bookshelf"
    },
    "correct_answer_key": "A",
    "gold_reasoning": "The answer is in the top half. The caption states the guitarist in the top half is next to a woman on a red couch. The 'brick wall' is a hallucination trap from the bottom half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  },
  {
    "question": "What kind of hair does the guitarist in the bottom half have?",
    "options": {
      "A": "Long and blonde",
      "B": "Curly hair and a beard",
      "C": "Short hair",
      "D": "A ponytail"
    },
    "correct_answer_key": "C",
    "gold_reasoning": "The answer is in the bottom half. The caption specifies the guitarist in the bottom half has short hair. 'Curly hair and a beard' is a hallucination trap from the top half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  }
]
```
---
**EXAMPLE 2**

**Caption:** "The video shows two violinists in different settings. The top frame features a violinist in a formal black suit playing in a grand concert hall. The bottom frame shows a violinist in a casual blue t-shirt playing outdoors in a park."

**Generated JSON:**
```json
[
  {
    "question": "What is the attire of the violinist playing in the concert hall?",
    "options": {
      "A": "A casual blue t-shirt",
      "B": "A formal black suit",
      "C": "A white tuxedo",
      "D": "A red dress"
    },
    "correct_answer_key": "B",
    "gold_reasoning": "The answer is in the top half. The caption describes the violinist in the concert hall (top) as wearing a formal black suit. The 'casual blue t-shirt' is a trap from the bottom half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  },
  {
    "question": "What is the setting for the violinist shown in the bottom frame?",
    "options": {
      "A": "A grand concert hall",
      "B": "A small, intimate studio",
      "C": "Outdoors in a park",
      "D": "On a balcony overlooking the sea"
    },
    "correct_answer_key": "C",
    "gold_reasoning": "The answer is in the bottom half. The caption specifies the setting for the violinist in the bottom frame is a park. The 'grand concert hall' is a trap from the top half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  }
]
```
---
**EXAMPLE 3**

**Caption:** "The video captures a split-screen of a choir performance. The top half shows a choir in black robes, accompanied by a group of violinists. The bottom half shows a choir in white robes, accompanied by a pianist."

**Generated JSON:**
```json
[
  {
    "question": "What instrument accompanies the choir shown in the top half of the video?",
    "options": {
      "A": "A piano",
      "B": "Violins",
      "C": "A harp",
      "D": "An organ"
    },
    "correct_answer_key": "B",
    "gold_reasoning": "The answer is in the top half. The caption states the choir in the top half is accompanied by violinists. The 'piano' is a hallucination trap from the bottom half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  },
  {
    "question": "What color are the robes worn by the choir that is accompanied by a pianist?",
    "options": {
      "A": "Black",
      "B": "Blue",
      "C": "White",
      "D": "Red"
    },
    "correct_answer_key": "C",
    "gold_reasoning": "The answer is in the bottom half. The caption specifies the choir accompanied by the pianist is wearing white robes. 'Black' robes are a hallucination trap from the top half.",
    "video_id": "placeholder_id",
    "category": "implicit distractions"
  }
]
```
"""

USER_PROMPT_TMPL = """
Video ID: {vid}

--- TOP half captions ---
Visual:
\"\"\"{top_vis}\"\"\"
Audio:
\"\"\"{top_aud}\"\"\"

--- BOTTOM half captions ---
Visual:
\"\"\"{bot_vis}\"\"\"
Audio:
\"\"\"{bot_aud}\"\"\"

Generate TWO MCQs now following all rules.
"""

def normalise(txt:str)->str:
    txt=re.sub(r"```(?:json)?|```","",txt).strip()
    txt=re.sub(r",\s*}","}",txt); txt=re.sub(r",\s*]","]",txt)
    return txt

@backoff.on_exception(backoff.expo, openai.OpenAIError, max_tries=MAX_RETRIES)
def gpt(client, model:str, temp:float, system:str, user:str)->str:
    r = client.chat.completions.create(
        model       = model,
        temperature = temp,
        messages = [
            {"role": "system", "content": system},
            {"role": "user",   "content": user}
        ]
    )
    return r.choices[0].message.content

def validate(item:Dict[str,Any],vid:str):
    req={"question","options","correct_answer_key","gold_reasoning",
         "video_id","category"}
    if not req.issubset(item): raise ValueError(f"keys {req-item.keys()}")
    item["video_id"]=vid
    item["category"]="implicit_distractions"

def read_txt(p:str)->str:
    try:  return Path(p).read_text(encoding='utf-8').strip()
    except: return ""

def get_processed_ids(output_path: Path, resume: bool) -> set:
    """Get IDs that have already been processed."""
    done = set()
    if output_path.exists() and resume:
        with open(output_path, encoding='utf-8') as f:
            for ln in f:
                try: done.add(json.loads(ln)["video_id"])
                except: pass
    return done

def main():
    # Parse arguments
    parser = argparse.ArgumentParser(
        description="Generate Implicit Distractions MCQs using GPT-4o",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument(
        "--data-dir",
        type=str,
        default=DEFAULT_CONFIG["data_dir"],
        help=f"Path to data directory (default: {DEFAULT_CONFIG['data_dir']})"
    )
    
    parser.add_argument(
        "--output-file",
        type=str,
        default=DEFAULT_CONFIG["output_file"],
        help=f"Output filename (default: {DEFAULT_CONFIG['output_file']})"
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
        "--no-resume",
        action="store_false",
        dest="resume",
        help="Start fresh, don't resume from previous run"
    )
    
    args = parser.parse_args()
    
    # Setup paths
    order_csv = Path(args.data_dir) / "order_log.csv"
    out_path = Path(args.output_file)
    
    # Initialize OpenAI client
    import os
    api_key = args.api_key or os.getenv("OPENAI_API_KEY")
    if not api_key:
        print("OPENAI_API_KEY not set or use --api-key")
        sys.exit(1)
        
    try:
        client = openai.Client(api_key=api_key)
    except openai.OpenAIError as e:
        print(f"OpenAI client error: {e}")
        sys.exit(1)
    
    # ----- load CSV -------------------------------------------------------
    rows=[]
    with open(order_csv,newline='',encoding='utf-8') as f:
        reader=csv.DictReader(f)
        for r in reader:
            rows.append(r)
    
    if not rows:
        print(f"No rows in {order_csv}"); sys.exit()
    
    # keys detection (expects these column names)
    VIS_TOP_KEY  = next(k for k in rows[0] if "top_visual"  in k.lower())
    AUD_TOP_KEY  = next(k for k in rows[0] if "top_audio"   in k.lower())
    VIS_BOT_KEY  = next(k for k in rows[0] if "bottom_visual" in k.lower())
    AUD_BOT_KEY  = next(k for k in rows[0] if "bottom_audio"  in k.lower())
    VID_KEY      = next(k for k in rows[0] if "video_name" in k.lower())
    
    # ----- skip already‑done vids ----------------------------------------
    done = get_processed_ids(out_path, args.resume)
    if done:
        print(f"{len(done)} already processed; skipping.")
    
    with open(out_path,"a",encoding="utf-8") as fout:
        for r in tqdm(rows,desc="Generating QA pairs",unit="clip"):
            vid=r[VID_KEY]
            if vid in done: continue
    
            top_vis=read_txt(r[VIS_TOP_KEY]);  bot_vis=read_txt(r[VIS_BOT_KEY])
            top_aud=read_txt(r[AUD_TOP_KEY]);  bot_aud=read_txt(r[AUD_BOT_KEY])
    
            prompt=USER_PROMPT_TMPL.format(
                vid=vid, top_vis=top_vis, top_aud=top_aud,
                bot_vis=bot_vis, bot_aud=bot_aud)
    
            try:
                raw=gpt(client, args.model, args.temperature, SYS_PROMPT, prompt)
                items=json.loads(normalise(raw))
                if not isinstance(items,list) or len(items)!=2:
                    raise ValueError("expected list len=2")
            except Exception as e:
                tqdm.write(f"ERROR for {vid} (API/Parse): {e}")
                continue
    
            try:
                for it in items:
                    validate(it,vid)
                    fout.write(json.dumps(it,ensure_ascii=False)+"\n")
                fout.flush()
                time.sleep(args.sleep_between)
            except Exception as e:
                tqdm.write(f"{vid}: validation error {e}")
    
    print(f"\nAll questions appended to {out_path}")

if __name__ == "__main__":
    main()
