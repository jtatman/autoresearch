"""
Fixed infrastructure for SmolLM-xLAM function-calling experiment.

Responsibilities (set once, never changed):
  - Download and cache a fixed 100-example eval set from Salesforce/xlam-function-calling-60k
  - Load the SmolLM-360M-Instruct-xLAM tokenizer
  - Format prompts in the xLAM chat-template style
  - Evaluate model function-calling accuracy (function name exact match)
  - Provide SFT batch formatter for train.py

Usage (one-time setup):
    python prepare.py

DO NOT MODIFY THIS FILE.
"""

import json
import math
import os

import torch

# ---------------------------------------------------------------------------
# Constants (fixed, do not modify)
# ---------------------------------------------------------------------------

MODEL_NAME    = "ericlewis/SmolLM-360M-Instruct-xLAM"
TIME_BUDGET   = 300          # training seconds per run (5 minutes)
EVAL_SIZE     = 25           # eval examples per run (subset of cached 100)
EVAL_DATASET  = "Salesforce/xlam-function-calling-60k"
EVAL_SPLIT_START = 59900     # 100 cached; EVAL_SIZE controls how many we use

CACHE_DIR     = os.path.join(os.path.expanduser("~"), ".cache", "smollm-xlam")
EVAL_CACHE    = os.path.join(CACHE_DIR, "eval_data.json")
LORA_BEST_DIR = os.path.join(CACHE_DIR, "best_lora")
MAX_SEQ_LEN   = 512          # max tokens for SFT examples

# ---------------------------------------------------------------------------
# xLAM prompt formatting
# ---------------------------------------------------------------------------

SYSTEM_PROMPT = (
    "You are an expert in composing functions. You are given a question and a "
    "set of possible functions. Based on the question, you will need to make one "
    "or more function/tool calls to achieve the purpose. If none of the functions "
    "can be used, point it out and refuse to answer. If the given question lacks "
    "the parameters required by the function, also point it out."
)

FORMAT_INSTRUCTION = (
    "The output MUST strictly adhere to the following JSON format, and NO other "
    "text MUST be included.\nThe example format is as follows. Please make sure "
    "the parameter type is correct. If no function call is needed, please make "
    "the tool calls an empty list '[]'.\n"
    '[{"name": "func_name1", "arguments": {"argument1": "value1", '
    '"argument2": "value2"}}, ... (more tool calls as required)]'
)


def format_xLAM_prompt(tools: list, query: str) -> str:
    """Format a single xLAM-style prompt (user turn only, no assistant response)."""
    tools_str = json.dumps(tools, ensure_ascii=False)
    return (
        f"[BEGIN OF TASK INSTRUCTION]\n"
        f"In this environment you have access to a set of tools you can use to "
        f"answer the user's question.\n"
        f"[END OF TASK INSTRUCTION]\n\n"
        f"[BEGIN OF AVAILABLE TOOLS]\n{tools_str}\n[END OF AVAILABLE TOOLS]\n\n"
        f"[BEGIN OF FORMAT INSTRUCTION]\n{FORMAT_INSTRUCTION}\n[END OF FORMAT INSTRUCTION]\n\n"
        f"[BEGIN OF QUERY]\nUser Query: {query}\n[END OF QUERY]"
    )


def build_chat_messages(tools: list, query: str) -> list:
    """Return messages list for tokenizer.apply_chat_template."""
    return [
        {"role": "system", "content": SYSTEM_PROMPT},
        {"role": "user",   "content": format_xLAM_prompt(tools, query)},
    ]

# ---------------------------------------------------------------------------
# Eval data: download once, cache as JSON
# ---------------------------------------------------------------------------

def download_eval_data() -> list:
    """Download fixed 100-example eval set and cache to disk. Returns list of dicts."""
    if os.path.exists(EVAL_CACHE):
        with open(EVAL_CACHE) as f:
            return json.load(f)

    print(f"Downloading eval data from {EVAL_DATASET} ...")
    from datasets import load_dataset
    ds = load_dataset(EVAL_DATASET, split="train")
    examples = []
    for i in range(EVAL_SPLIT_START, EVAL_SPLIT_START + EVAL_SIZE):
        row = ds[i]
        # Each row: tools (JSON string), query (str), answers (JSON string)
        try:
            tools   = json.loads(row["tools"])   if isinstance(row["tools"],   str) else row["tools"]
            answers = json.loads(row["answers"]) if isinstance(row["answers"], str) else row["answers"]
        except (json.JSONDecodeError, KeyError):
            continue
        examples.append({
            "tools":   tools,
            "query":   row["query"],
            "answers": answers,   # list of {"name": ..., "arguments": ...}
        })

    os.makedirs(CACHE_DIR, exist_ok=True)
    with open(EVAL_CACHE, "w") as f:
        json.dump(examples, f, ensure_ascii=False, indent=2)
    print(f"Cached {len(examples)} eval examples to {EVAL_CACHE}")
    return examples

# ---------------------------------------------------------------------------
# Tokenizer
# ---------------------------------------------------------------------------

def load_tokenizer():
    """Load (and cache) the SmolLM-xLAM tokenizer."""
    from transformers import AutoTokenizer
    tok = AutoTokenizer.from_pretrained(MODEL_NAME, cache_dir=CACHE_DIR)
    if tok.pad_token is None:
        tok.pad_token = tok.eos_token
    return tok

# ---------------------------------------------------------------------------
# Evaluation (DO NOT CHANGE — fixed metric)
# ---------------------------------------------------------------------------

@torch.no_grad()
def evaluate_function_calling(model, tokenizer, batch_size: int = 8, device: str = "cuda") -> dict:
    """
    Evaluate function-calling accuracy on the fixed eval set.

    Primary metric: function name exact match (fraction of examples where the
    model's first predicted function name matches the ground-truth first function name).

    Returns dict: {"name_accuracy": float, "parse_rate": float, "n": int}
    """
    examples = download_eval_data()[:EVAL_SIZE]
    model.eval()

    n_correct = 0
    n_parseable = 0
    n_total = len(examples)

    for start in range(0, n_total, batch_size):
        batch = examples[start:start + batch_size]
        prompts = []
        for ex in batch:
            msgs = build_chat_messages(ex["tools"], ex["query"])
            prompt = tokenizer.apply_chat_template(
                msgs, tokenize=False, add_generation_prompt=True
            )
            prompts.append(prompt)

        inputs = tokenizer(
            prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=MAX_SEQ_LEN,
        ).to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out_ids = model.generate(
                **inputs,
                max_new_tokens=128,
                do_sample=False,
                pad_token_id=tokenizer.pad_token_id,
            )

        input_len = inputs["input_ids"].shape[1]
        for i, ex in enumerate(batch):
            gen_ids = out_ids[i, input_len:]
            raw = tokenizer.decode(gen_ids, skip_special_tokens=True).strip()

            # Try to parse the JSON output
            try:
                # Strip markdown fences
                text = raw
                if "```" in text:
                    text = text.split("```")[1]
                    if text.startswith("json"):
                        text = text[4:]
                text = text.strip()
                # Strip any prose preamble before the first [ or { character
                bracket = text.find("[")
                brace   = text.find("{")
                if bracket >= 0 or brace >= 0:
                    start = bracket if brace < 0 else (brace if bracket < 0 else min(bracket, brace))
                    text = text[start:]
                calls = json.loads(text)
                if not isinstance(calls, list):
                    calls = [calls]
                calls = [c for c in calls if isinstance(c, dict)]
                n_parseable += 1

                # Compare first function name
                gt_names = [a["name"] for a in ex["answers"] if isinstance(a, dict) and "name" in a]
                pred_names = [c.get("name", "") for c in calls]
                if gt_names and pred_names and gt_names[0] == pred_names[0]:
                    n_correct += 1
            except (json.JSONDecodeError, KeyError, IndexError):
                pass  # unparseable output counts as wrong

    return {
        "name_accuracy": n_correct / n_total if n_total else 0.0,
        "parse_rate":    n_parseable / n_total if n_total else 0.0,
        "n":             n_total,
    }

# ---------------------------------------------------------------------------
# SFT batch formatter (used by train.py)
# ---------------------------------------------------------------------------

def make_sft_batch(pairs: list[tuple[str, str]], tokenizer, max_length: int = MAX_SEQ_LEN):
    """
    Tokenize (prompt, response) pairs for causal-LM SFT.

    Loss is computed only on the response tokens (prompt tokens are masked to -100).

    Args:
        pairs: list of (prompt_str, response_str) — both already formatted as plain text.
               The prompt should be the full chat-formatted user turn (use
               tokenizer.apply_chat_template). The response is the assistant's JSON reply.
        tokenizer: loaded tokenizer
        max_length: max sequence length (truncate silently)

    Returns:
        dict with "input_ids", "attention_mask", "labels" — all torch.LongTensor, padded.
    """
    all_input_ids = []
    all_labels    = []

    for prompt, response in pairs:
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)

        seq = p_ids + r_ids
        if len(seq) > max_length:
            # Truncate from the left of the prompt to preserve the response
            overflow = len(seq) - max_length
            p_ids = p_ids[overflow:]
            seq = p_ids + r_ids

        labels = [-100] * len(p_ids) + r_ids  # mask prompt tokens

        all_input_ids.append(seq)
        all_labels.append(labels)

    # Pad to longest in batch
    max_len = max(len(s) for s in all_input_ids)
    pad_id  = tokenizer.pad_token_id

    input_ids_t  = torch.full((len(all_input_ids), max_len), pad_id,  dtype=torch.long)
    attention_t  = torch.zeros((len(all_input_ids), max_len),          dtype=torch.long)
    labels_t     = torch.full((len(all_labels),    max_len), -100,    dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(all_input_ids, all_labels)):
        input_ids_t[i, :len(ids)]  = torch.tensor(ids,  dtype=torch.long)
        attention_t[i,  :len(ids)] = 1
        labels_t[i,    :len(labs)] = torch.tensor(labs, dtype=torch.long)

    return {
        "input_ids":      input_ids_t,
        "attention_mask": attention_t,
        "labels":         labels_t,
    }

# ---------------------------------------------------------------------------
# Main: one-time setup
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print(f"Cache directory: {CACHE_DIR}")
    print()
    print("Step 1: downloading tokenizer ...")
    tok = load_tokenizer()
    print(f"  vocab_size={tok.vocab_size}, pad='{tok.pad_token}'")
    print()
    print("Step 2: downloading eval data ...")
    examples = download_eval_data()
    print(f"  {len(examples)} eval examples ready.")
    print()
    print("Done! Ready to run train.py.")
