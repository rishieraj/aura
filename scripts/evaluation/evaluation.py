#!/usr/bin/env python3
"""
Evaluate model responses for multi-modal QA benchmark using GPT-4o and NLI models.
"""

import os, sys, re
import json
import time
import argparse
from pathlib import Path
from typing import Dict, Any

import backoff
import openai
from tqdm import tqdm
import torch
from sentence_transformers.cross_encoder import CrossEncoder

# Default config
DEFAULT_CONFIG = {
    "llm_model": "gpt-4o",
    "nli_model": "cross-encoder/nli-deberta-v3-base",
    "temperature": 0.1,
    "sleep_between": 1.0
}

MAX_RETRIES = 5

# ─── Prompts ─────────────────────────────────────────────────────────────────

ANSWER_CHECK_SYSTEM_PROMPT = """
You are a precise AI evaluator for multiple-choice questions. Your task is to determine if a `Generated Answer` is semantically equivalent to the `Correct Answer`.

**Core Rules:**
1.  **Ignore the Option Letter:** Completely disregard the leading letter (e.g., "A.", "B.", "C."). The letters can be different.
2.  **Focus on Meaning:** The answer is correct if it conveys the same essential information. Minor differences in wording (e.g., "wearing a black shirt" vs. "has a black shirt") should be considered correct.
3.  **Strict on Key Facts:** Be strict about key factual differences (e.g., "black shirt" vs. "blue shirt" is incorrect).

**Output Format:** Return a single JSON object with one key, `is_correct`, which is a boolean (`true` or `false`).
---
**EXAMPLE 1:**
- Generated Answer: "C. The man has a black shirt"
- Correct Answer: "B. The man is wearing a black shirt"
- Output: `{"is_correct": true}`

**EXAMPLE 2:**
- Generated Answer: "A. A red car"
- Correct Answer: "D. A blue car"
- Output: `{"is_correct": false}`

**EXAMPLE 3:**
- Generated Answer: "A black shirt and white pants"
- Correct Answer: "A. A dark polka-dotted shirt and white pants"
- Output: `{"is_correct": false}`
---
"""

ANSWER_CHECK_USER_PROMPT_TMPL = """\
Generated Answer:
"{generated}"

Correct Answer:
"{correct}"

Is the generated answer semantically equivalent to the correct answer?
"""

FACTUAL_SYSTEM_PROMPT = """
You are a Meticulous Fact-Checker AI. Your task is to compare a "Generated Reasoning" statement against a "Ground Truth (GT) Reasoning" statement and produce a Factual Consistency Score.

**Core Task:** Evaluate how well the facts in the Generated Reasoning align with the facts in the GT Reasoning. The score should reflect the proportion of matching facts.

**Instructions:**
1.  **Deconstruct Both Statements:** For each statement, break it down into its core factual entities. Identify key entities like `subject`, `object`, `attribute`, `action`, and `location`.
2.  **Compare Fact-by-Fact:** Systematically compare the deconstructed facts from the Generated Reasoning to those from the GT Reasoning.
3.  **Calculate Score:** Compute a `factual_consistency_score` from 0.0 to 1.0. The score is the fraction of facts from the GT Reasoning that are correctly stated in the Generated Reasoning.
4.  **Provide Explanation:** In the `explanation` field, clearly list which facts matched and which were mismatched or missing.
5.  **Output Format:** Return **one single JSON object** and nothing else. The JSON must contain `factual_consistency_score` (a float) and `explanation` (a string).

---
**EXAMPLE 1:**

**Inputs:**
- Generated Reasoning: "The man playing the violin on the left side produces the highest pitch of music."
- GT Reasoning: "The woman playing the violin on the right side has the highest pitch instrument."

**Analysis Steps (Internal Thought Process):**
1.  **Deconstruct GT Reasoning:**
    - `subject`: "woman"
    - `object`: "violin"
    - `location`: "right side"
    - `attribute`: "highest pitch"
2.  **Deconstruct Generated Reasoning:**
    - `subject`: "man"
    - `object`: "violin"
    - `location`: "left side"
    - `attribute`: "highest pitch"
3.  **Compare:**
    - `subject`: Mismatch ("man" vs "woman")
    - `object`: Match ("violin")
    - `location`: Mismatch ("left side" vs "right side")
    - `attribute`: Match ("highest pitch")
4.  **Score:** 2 out of 4 facts match. Score = 0.5.

**Generated JSON:**
```json
{
  "factual_consistency_score": 0.5,
  "explanation": "The Generated Reasoning correctly identified the 'object' (violin) and 'attribute' (highest pitch). However, it failed on the 'subject' (stating 'man' instead of 'woman') and the 'location' (stating 'left side' instead of 'right side')."
}
```
---
**EXAMPLE 2:**

**Inputs:**
- Generated Reasoning: "The blue car is driving quickly."
- GT Reasoning: "The blue car is moving fast."

**Analysis Steps (Internal Thought Process):**
1.  **Deconstruct GT Reasoning:**
    - `object`: "blue car"
    - `action`: "moving fast"
2.  **Deconstruct Generated Reasoning:**
    - `object`: "blue car"
    - `action`: "driving quickly"
3.  **Compare:**
    - `object`: Match ("blue car")
    - `action`: Match (semantically equivalent)
4.  **Score:** 2 out of 2 facts match. Score = 1.0.

**Generated JSON:**
```json
{
  "factual_consistency_score": 1.0,
  "explanation": "All facts match. The Generated Reasoning correctly identified the 'object' (blue car) and the 'action' (driving quickly is semantically equivalent to moving fast)."
}
```
---
"""

FACTUAL_USER_PROMPT_TMPL = """\
Generated Reasoning:
"{generated}"

GT Reasoning:
"{gt}"

Generate a JSON object with the factual consistency score and an explanation.
"""

SANITIZER_SYSTEM_PROMPT = """
You are a specialist in Logical Abstraction. Your task is to distill a detailed reasoning statement into its abstract "Core Inferential Claim".

**Core Rule:** The goal is to remove ALL specific, grounded details (like who, where, what color, or what specific object) and keep ONLY the central logical relationship being expressed. The output should be as abstract as possible.

**Instructions:**
1.  **Identify the Core Claim:** Read the input sentence and identify the main logical point. What is being related to what?
2.  **Remove Specific Entities & Attributes:** Eliminate specific subjects (e.g., 'the man'), locations (e.g., 'in the top half'), and descriptive attributes (e.g., 'polka-dotted', 'white pants', 'marimba') unless the attribute itself is the entire point of the reasoning.
3.  **Generalize the Statement:** Rephrase the claim using generic placeholders (e.g., 'the subject', 'the object', 'the setting', 'an attribute', 'an instrument'). The output must be a concise, abstract statement.
4.  **Focus on the Relationship:** Preserve the core relationship, such as possession of an attribute, a causal link, or a comparison.
5.  **Output Format:** Return **one single JSON object** with a single key, `sanitized_reasoning`, containing the abstract claim as a string.

---
**EXAMPLE 1 (Focus on Attribute):**

**Input Sentence:** "The visual caption describes the flutist in the top half as wearing a black shirt with a polka dot pattern."
**Analysis:** The core idea is that a subject's attire has a certain pattern. The subject, location, and color are details. The pattern is the key attribute.
**Generated JSON:**
```json
{
  "sanitized_reasoning": "The subject's attire has a specific pattern."
}
```
---
**EXAMPLE 2 (Focus on Object):**

**Input Sentence:** "The answer is in the bottom half. The visual caption specifies the man is playing a marimba in the bottom half."
**Analysis:** The core idea is that a person is playing a specific type of instrument. The location and gender are details.
**Generated JSON:**
```json
{
  "sanitized_reasoning": "The subject is playing a specific instrument."
}
```
---
**EXAMPLE 3 (Focus on Setting):**

**Input Sentence:** "The answer is in the bottom half. The visual caption describes the setting as a dimly lit hall during the Jerusalem International Chamber Festival."
**Analysis:** The core idea is the description of the setting's lighting. The specific festival and location are details.
**Generated JSON:**
```json
{
  "sanitized_reasoning": "The setting is described as dimly lit."
}
```
---
**EXAMPLE 4 (Stricter Abstraction):**

**Input Sentence:** "He is clearly wearing a black shirt and white pants while playing the flute in the top half of the video."
**Analysis:** The core idea is that a subject is wearing specific attire while performing an action. All details (color, clothing items, instrument, location) must be abstracted.
**Generated JSON:**
```json
{
  "sanitized_reasoning": "The subject is wearing specific attire while performing an action."
}
```
---
"""

SANITIZER_USER_PROMPT_TMPL = """\
Reasoning Statement:
"{reasoning_string}"

Generate a JSON object containing the sanitized "Core Inferential Claim".
"""

# ─── Helper functions ────────────────────────────────────────────────────────

def normalise_json_str(txt: str) -> str:
    txt = txt.strip()
    txt = re.sub(r"^```(json)?\s*|\s*```$", "", txt, flags=re.MULTILINE)
    txt = re.sub(r",\s*}", "}", txt)
    txt = re.sub(r",\s*]", "]", txt)
    return txt

@backoff.on_exception(backoff.expo, openai.RateLimitError, max_tries=MAX_RETRIES)
def call_llm(client, model: str, temp: float, system_prompt: str, user_prompt: str) -> Dict[str, Any]:
    resp = client.chat.completions.create(
        model=model,
        temperature=temp,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": user_prompt}
        ],
        response_format={"type": "json_object"}
    )
    clean = normalise_json_str(resp.choices[0].message.content)
    return json.loads(clean)

def get_processed_ids(output_path: Path) -> set:
    processed_ids = set()
    if output_path.exists():
        with output_path.open("r", encoding="utf-8") as f:
            for line in f:
                try:
                    rec = json.loads(line)
                    processed_ids.add(rec.get('question'))
                except:
                    pass
    return processed_ids

# ─── Main ────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="Evaluate model responses for multi-modal QA benchmark",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    
    parser.add_argument("--input", type=str, required=True,
                       help="Input JSONL file with model responses")
    parser.add_argument("--output", type=str, required=True,
                       help="Output JSONL file for evaluation results")
    parser.add_argument("--model-key", type=str, required=True,
                       help="Model identifier key in JSON (e.g., 'gemini_2.0_flash')")
    parser.add_argument("--api-key", type=str, help="OpenAI API key")
    parser.add_argument("--llm-model", type=str, default=DEFAULT_CONFIG["llm_model"],
                       help=f"LLM model for evaluation (default: {DEFAULT_CONFIG['llm_model']})")
    parser.add_argument("--nli-model", type=str, default=DEFAULT_CONFIG["nli_model"],
                       help=f"NLI model for inference scoring (default: {DEFAULT_CONFIG['nli_model']})")
    parser.add_argument("--temperature", type=float, default=DEFAULT_CONFIG["temperature"],
                       help=f"Temperature for LLM (default: {DEFAULT_CONFIG['temperature']})")
    parser.add_argument("--sleep-between", type=float, default=DEFAULT_CONFIG["sleep_between"],
                       help=f"Sleep between API calls (default: {DEFAULT_CONFIG['sleep_between']})")
    parser.add_argument("--no-resume", action="store_true",
                       help="Start fresh, ignore previous progress")
    
    args = parser.parse_args()
    
    input_file = Path(args.input)
    output_file = Path(args.output)
    
    if not input_file.exists():
        print(f"Error: Input file not found at '{input_file}'")
        sys.exit(1)
    
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
    
    # Load records
    all_records = [json.loads(line) for line in input_file.open("r", encoding="utf-8")]
    
    # Handle resume logic
    processed_records = []
    if not args.no_resume and output_file.exists():
        processed_records = [json.loads(line) for line in output_file.open("r", encoding="utf-8")]
    
    processed_ids = {rec.get('question') for rec in processed_records}
    records_to_process = [rec for rec in all_records if rec.get('question') not in processed_ids]
    
    print(f"Found {len(all_records)} total records")
    if processed_records:
        print(f"{len(processed_records)} records already processed. Skipping.")
    print(f"Processing {len(records_to_process)} new records")
    
    if not records_to_process:
        print("No new records to process")
    else:
        # Load NLI model
        device = "cuda" if torch.cuda.is_available() else "cpu"
        print(f"Loading NLI model '{args.nli_model}' on {device}")
        nli_model = CrossEncoder(args.nli_model, device=device)
        
        model_answer_key = f"{args.model_key}_answer"
        model_reason_key = f"{args.model_key}_reason"
        
        for item in tqdm(records_to_process, desc="Evaluating", unit="item"):
            try:
                correct_answer_key = item.get("correct_answer_key")
                correct_answer_text = item.get("options", {}).get(correct_answer_key)
                generated_answer_text = item.get(model_answer_key)
                gt_reasoning = item.get("gold_reasoning")
                generated_reasoning = item.get(model_reason_key)
                
                if not all([correct_answer_key, correct_answer_text, generated_answer_text, 
                           gt_reasoning, generated_reasoning]):
                    tqdm.write(f"Skipping item for video {item.get('video_id')} - missing data")
                    continue
                
                # 1. Check answer correctness
                answer_eval = call_llm(client, args.llm_model, args.temperature,
                                      ANSWER_CHECK_SYSTEM_PROMPT, 
                                      ANSWER_CHECK_USER_PROMPT_TMPL.format(
                                          generated=generated_answer_text,
                                          correct=correct_answer_text))
                is_correct = answer_eval.get("is_correct", False)
                item["answer_correctness"] = {"is_correct": is_correct}
                time.sleep(args.sleep_between)
                
                # 2. Conditional reasoning evaluation
                if is_correct:
                    # Factual consistency
                    factual_eval = call_llm(client, args.llm_model, args.temperature,
                                           FACTUAL_SYSTEM_PROMPT,
                                           FACTUAL_USER_PROMPT_TMPL.format(
                                               generated=generated_reasoning,
                                               gt=gt_reasoning))
                    item["factual_consistency_evaluation"] = factual_eval
                    time.sleep(args.sleep_between)
                    
                    # Core inference
                    sanitized_gt = call_llm(client, args.llm_model, args.temperature,
                                           SANITIZER_SYSTEM_PROMPT,
                                           SANITIZER_USER_PROMPT_TMPL.format(
                                               reasoning_string=gt_reasoning))
                    sanitized_gt_text = sanitized_gt.get("sanitized_reasoning", "")
                    time.sleep(args.sleep_between)
                    
                    sanitized_gen = call_llm(client, args.llm_model, args.temperature,
                                            SANITIZER_SYSTEM_PROMPT,
                                            SANITIZER_USER_PROMPT_TMPL.format(
                                                reasoning_string=generated_reasoning))
                    sanitized_gen_text = sanitized_gen.get("sanitized_reasoning", "")
                    
                    nli_scores = nli_model.predict([(sanitized_gt_text, sanitized_gen_text)], 
                                                  apply_softmax=True)[0]
                    item["core_inference_evaluation"] = {
                        "sanitized_gold_reasoning": sanitized_gt_text,
                        "sanitized_generated_reasoning": sanitized_gen_text,
                        "core_inference_score": float(nli_scores[1]),
                    }
                else:
                    # Answer wrong - assign 0 scores
                    item["factual_consistency_evaluation"] = {
                        "factual_consistency_score": 0.0,
                        "explanation": "Answer was incorrect; not evaluated."
                    }
                    item["core_inference_evaluation"] = {
                        "core_inference_score": 0.0,
                        "explanation": "Answer was incorrect; not evaluated."
                    }
                
                processed_records.append(item)
                
            except Exception as e:
                tqdm.write(f"ERROR processing item for video {item.get('video_id', 'N/A')}: {e}")
                continue
        
        # Write all records
        with output_file.open("w", encoding="utf-8") as out_f:
            for rec in processed_records:
                out_f.write(json.dumps(rec, ensure_ascii=False) + "\n")
    
    # Final statistics
    if processed_records:
        answer_scores = [1.0 if rec.get("answer_correctness", {}).get("is_correct") else 0.0 
                        for rec in processed_records]
        factual_scores = [rec.get("factual_consistency_evaluation", {}).get("factual_consistency_score", 0.0) 
                         for rec in processed_records]
        inference_scores = [rec.get("core_inference_evaluation", {}).get("core_inference_score", 0.0) 
                           for rec in processed_records]
        
        avg_answer = (sum(answer_scores) / len(answer_scores)) * 100
        avg_factual = (sum(factual_scores) / len(factual_scores)) * 100
        avg_inference = (sum(inference_scores) / len(inference_scores)) * 100
        
        print("\n--- Final Benchmark Averages ---")
        print(f"Total Questions Evaluated : {len(processed_records)}")
        print(f"Answer Correctness        : {avg_answer:.2f}%")
        print(f"Factual Consistency Score : {avg_factual:.2f}%")
        print(f"Core Inference Score      : {avg_inference:.2f}%")
        print("----------------------------------")
    else:
        print("No evaluation data to report")

if __name__ == "__main__":
    main()