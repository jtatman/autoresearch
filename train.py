"""
SmolLM-xLAM function-calling experiment. Single-GPU, LoRA fine-tuning.
Usage: uv run train.py

This file changes every run. Edit:
  - TRAINING_PAIRS: the 30 (prompt, response) examples used to fine-tune
  - LoRA / optimizer hyperparameters below
  - Everything else is fair game as long as it runs in TIME_BUDGET seconds

The fixed metric is evaluate_function_calling() in prepare.py. Goal: increase
name_accuracy by >= 0.01 across runs.
"""

import json
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import LoraConfig, TaskType, get_peft_model, PeftModel
from transformers import AutoModelForCausalLM

from prepare import (
    MODEL_NAME, TIME_BUDGET, CACHE_DIR, LORA_BEST_DIR,
    build_chat_messages, evaluate_function_calling, load_tokenizer, make_sft_batch,
)

# ---------------------------------------------------------------------------
# Hyperparameters (edit freely each run)
# ---------------------------------------------------------------------------

USE_8BIT       = False   # GTX 1070 is sm_61 — bitsandbytes int8 requires sm_70+
LORA_R         = 16      # LoRA rank
LORA_ALPHA     = 32      # LoRA scaling (alpha/r = 2)
LORA_DROPOUT   = 0.05
LORA_TARGETS   = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE  = 2e-4
WEIGHT_DECAY   = 0.01
EPOCHS         = 3       # passes over the 30 training pairs
MICRO_BATCH    = 4       # examples per gradient step
GRAD_ACCUM     = 2       # effective batch = MICRO_BATCH * GRAD_ACCUM = 8

EVAL_BATCH     = 4       # batch size for evaluate_function_calling

# ---------------------------------------------------------------------------
# 30 training pairs — CHANGE THESE EACH RUN
#
# Format: (prompt, response)
#   prompt   — full xLAM user turn (use build_prompt() helper below)
#   response — the JSON string the model should output
#
# Run 1: broad coverage baseline — weather, search, math, calendar, maps,
# finance, news, email, translation, unit conversion, timezone, code execution,
# image search, recipe, sports, movies, music, flights, hotels, crypto,
# reminders, jokes, dictionary, thesaurus, grammar check, PDF, QR code,
# password gen, URL shortener, IP lookup.
# ---------------------------------------------------------------------------

def build_prompt(tools: list, query: str) -> str:
    msgs = build_chat_messages(tools, query)
    from prepare import load_tokenizer as _tok
    # tokenizer is already loaded globally below; use it via closure
    return _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)


# Tool schema helpers
def _tool(name, desc, params: dict) -> dict:
    return {
        "name": name,
        "description": desc,
        "parameters": {
            "type": "object",
            "properties": {k: {"type": v[0], "description": v[1]} for k, v in params.items()},
            "required": [k for k, v in params.items() if len(v) > 2 and v[2]],
        },
    }


PAIR_SPECS = [
    # (tools_list, query, answer_list)
    (
        [_tool("get_weather", "Get current weather for a city",
               {"city": ("string", "City name", True), "unit": ("string", "celsius or fahrenheit", False)})],
        "What's the weather like in Tokyo right now?",
        [{"name": "get_weather", "arguments": {"city": "Tokyo"}}],
    ),
    (
        [_tool("web_search", "Search the web for information",
               {"query": ("string", "Search query", True), "num_results": ("integer", "Number of results", False)})],
        "Search the web for the latest news about large language models.",
        [{"name": "web_search", "arguments": {"query": "latest news about large language models"}}],
    ),
    (
        [_tool("calculator", "Evaluate a mathematical expression",
               {"expression": ("string", "Math expression to evaluate", True)})],
        "What is 347 multiplied by 29?",
        [{"name": "calculator", "arguments": {"expression": "347 * 29"}}],
    ),
    (
        [_tool("create_calendar_event", "Create a calendar event",
               {"title": ("string", "Event title", True),
                "date": ("string", "Date in YYYY-MM-DD format", True),
                "time": ("string", "Time in HH:MM format", False)})],
        "Schedule a dentist appointment for next Monday at 10am.",
        [{"name": "create_calendar_event", "arguments": {"title": "Dentist appointment", "date": "2025-05-05", "time": "10:00"}}],
    ),
    (
        [_tool("get_directions", "Get driving directions between two locations",
               {"origin": ("string", "Starting location", True),
                "destination": ("string", "Ending location", True),
                "mode": ("string", "Travel mode: driving, walking, transit", False)})],
        "How do I get from San Francisco to Los Angeles by car?",
        [{"name": "get_directions", "arguments": {"origin": "San Francisco", "destination": "Los Angeles", "mode": "driving"}}],
    ),
    (
        [_tool("get_stock_price", "Get the current stock price for a ticker symbol",
               {"ticker": ("string", "Stock ticker symbol", True)})],
        "What is Apple's current stock price?",
        [{"name": "get_stock_price", "arguments": {"ticker": "AAPL"}}],
    ),
    (
        [_tool("get_news", "Fetch top news headlines by topic",
               {"topic": ("string", "News topic", True), "country": ("string", "Country code e.g. us", False)})],
        "Get me the latest sports headlines in the US.",
        [{"name": "get_news", "arguments": {"topic": "sports", "country": "us"}}],
    ),
    (
        [_tool("send_email", "Send an email message",
               {"to": ("string", "Recipient email address", True),
                "subject": ("string", "Email subject", True),
                "body": ("string", "Email body", True)})],
        "Send an email to alice@example.com with subject 'Meeting tomorrow' saying the 3pm meeting is confirmed.",
        [{"name": "send_email", "arguments": {"to": "alice@example.com", "subject": "Meeting tomorrow", "body": "The 3pm meeting is confirmed."}}],
    ),
    (
        [_tool("translate_text", "Translate text to another language",
               {"text": ("string", "Text to translate", True),
                "target_language": ("string", "Target language name", True)})],
        "Translate 'Hello, how are you?' into French.",
        [{"name": "translate_text", "arguments": {"text": "Hello, how are you?", "target_language": "French"}}],
    ),
    (
        [_tool("convert_units", "Convert a value between units",
               {"value": ("number", "Value to convert", True),
                "from_unit": ("string", "Source unit", True),
                "to_unit": ("string", "Target unit", True)})],
        "Convert 100 miles to kilometers.",
        [{"name": "convert_units", "arguments": {"value": 100, "from_unit": "miles", "to_unit": "kilometers"}}],
    ),
    (
        [_tool("get_timezone", "Get the current time in a specified timezone",
               {"timezone": ("string", "Timezone name e.g. America/New_York", True)})],
        "What time is it right now in Sydney, Australia?",
        [{"name": "get_timezone", "arguments": {"timezone": "Australia/Sydney"}}],
    ),
    (
        [_tool("run_code", "Execute a code snippet and return output",
               {"code": ("string", "Code to execute", True),
                "language": ("string", "Programming language", True)})],
        "Run this Python code and tell me the output: print(sum(range(1, 101)))",
        [{"name": "run_code", "arguments": {"code": "print(sum(range(1, 101)))", "language": "python"}}],
    ),
    (
        [_tool("image_search", "Search for images on the web",
               {"query": ("string", "Image search query", True),
                "num_images": ("integer", "Number of images to return", False)})],
        "Find me pictures of golden retriever puppies.",
        [{"name": "image_search", "arguments": {"query": "golden retriever puppies"}}],
    ),
    (
        [_tool("get_recipe", "Search for a recipe by dish name",
               {"dish": ("string", "Dish name", True),
                "dietary": ("string", "Dietary restriction e.g. vegan, gluten-free", False)})],
        "Give me a vegan recipe for chocolate chip cookies.",
        [{"name": "get_recipe", "arguments": {"dish": "chocolate chip cookies", "dietary": "vegan"}}],
    ),
    (
        [_tool("get_sports_score", "Get live or recent score for a sports game",
               {"sport": ("string", "Sport type e.g. basketball, soccer", True),
                "team": ("string", "Team name", False)})],
        "What was the Lakers' score in their last game?",
        [{"name": "get_sports_score", "arguments": {"sport": "basketball", "team": "Lakers"}}],
    ),
    (
        [_tool("search_movies", "Search for movies by title or genre",
               {"query": ("string", "Movie title or description", True),
                "genre": ("string", "Genre filter", False)})],
        "Find me some good sci-fi movies from the 1980s.",
        [{"name": "search_movies", "arguments": {"query": "sci-fi movies from the 1980s", "genre": "sci-fi"}}],
    ),
    (
        [_tool("play_music", "Play a song or artist on the music player",
               {"query": ("string", "Song or artist name", True),
                "shuffle": ("boolean", "Shuffle the playlist", False)})],
        "Play some music by The Beatles.",
        [{"name": "play_music", "arguments": {"query": "The Beatles"}}],
    ),
    (
        [_tool("search_flights", "Search for available flights between two airports",
               {"origin": ("string", "Origin airport code", True),
                "destination": ("string", "Destination airport code", True),
                "date": ("string", "Departure date YYYY-MM-DD", True),
                "passengers": ("integer", "Number of passengers", False)})],
        "Find flights from New York (JFK) to London (LHR) on June 15th for 2 passengers.",
        [{"name": "search_flights", "arguments": {"origin": "JFK", "destination": "LHR", "date": "2025-06-15", "passengers": 2}}],
    ),
    (
        [_tool("search_hotels", "Search for hotels in a city",
               {"city": ("string", "City name", True),
                "checkin": ("string", "Check-in date YYYY-MM-DD", True),
                "checkout": ("string", "Check-out date YYYY-MM-DD", True),
                "guests": ("integer", "Number of guests", False)})],
        "Find hotels in Paris for 2 guests checking in July 10 and checking out July 15.",
        [{"name": "search_hotels", "arguments": {"city": "Paris", "checkin": "2025-07-10", "checkout": "2025-07-15", "guests": 2}}],
    ),
    (
        [_tool("get_crypto_price", "Get the current price of a cryptocurrency",
               {"symbol": ("string", "Crypto symbol e.g. BTC, ETH", True),
                "currency": ("string", "Fiat currency to show price in", False)})],
        "What is the current price of Ethereum in USD?",
        [{"name": "get_crypto_price", "arguments": {"symbol": "ETH", "currency": "USD"}}],
    ),
    (
        [_tool("set_reminder", "Set a reminder for a future time",
               {"message": ("string", "Reminder message", True),
                "datetime": ("string", "When to remind in ISO 8601 format", True)})],
        "Remind me to call mom tonight at 7pm.",
        [{"name": "set_reminder", "arguments": {"message": "Call mom", "datetime": "2025-05-03T19:00:00"}}],
    ),
    (
        [_tool("get_joke", "Fetch a random joke optionally filtered by category",
               {"category": ("string", "Joke category e.g. programming, general", False)})],
        "Tell me a programming joke.",
        [{"name": "get_joke", "arguments": {"category": "programming"}}],
    ),
    (
        [_tool("lookup_word", "Look up the definition of a word",
               {"word": ("string", "Word to look up", True),
                "language": ("string", "Language code e.g. en, es", False)})],
        "What does the word 'ephemeral' mean?",
        [{"name": "lookup_word", "arguments": {"word": "ephemeral"}}],
    ),
    (
        [_tool("get_synonyms", "Get synonyms for a word",
               {"word": ("string", "Word to find synonyms for", True)})],
        "Give me synonyms for the word 'happy'.",
        [{"name": "get_synonyms", "arguments": {"word": "happy"}}],
    ),
    (
        [_tool("check_grammar", "Check and correct grammar in a text passage",
               {"text": ("string", "Text to check", True)})],
        "Check the grammar in this sentence: 'She don't like apples'.",
        [{"name": "check_grammar", "arguments": {"text": "She don't like apples"}}],
    ),
    (
        [_tool("extract_pdf_text", "Extract text from a PDF file",
               {"file_path": ("string", "Path to the PDF file", True),
                "page_range": ("string", "Page range e.g. 1-5", False)})],
        "Extract text from the file at /home/user/report.pdf pages 1 through 3.",
        [{"name": "extract_pdf_text", "arguments": {"file_path": "/home/user/report.pdf", "page_range": "1-3"}}],
    ),
    (
        [_tool("generate_qr_code", "Generate a QR code for a URL or text",
               {"content": ("string", "Content to encode", True),
                "size": ("integer", "QR code size in pixels", False)})],
        "Generate a QR code for the URL https://example.com.",
        [{"name": "generate_qr_code", "arguments": {"content": "https://example.com"}}],
    ),
    (
        [_tool("generate_password", "Generate a secure random password",
               {"length": ("integer", "Password length", True),
                "include_symbols": ("boolean", "Include symbols in password", False)})],
        "Generate a 16-character password with symbols.",
        [{"name": "generate_password", "arguments": {"length": 16, "include_symbols": True}}],
    ),
    (
        [_tool("shorten_url", "Shorten a long URL",
               {"url": ("string", "URL to shorten", True)})],
        "Shorten this URL: https://www.example.com/very/long/path/to/some/page?query=value&other=123",
        [{"name": "shorten_url", "arguments": {"url": "https://www.example.com/very/long/path/to/some/page?query=value&other=123"}}],
    ),
    (
        [_tool("lookup_ip", "Look up information about an IP address",
               {"ip": ("string", "IP address to look up", True)})],
        "What country does the IP address 8.8.8.8 belong to?",
        [{"name": "lookup_ip", "arguments": {"ip": "8.8.8.8"}}],
    ),
]

assert len(PAIR_SPECS) == 30, f"Expected 30 training pairs, got {len(PAIR_SPECS)}"

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
device  = torch.device("cuda")

print(f"Loading tokenizer from {MODEL_NAME} ...")
_TOKENIZER = load_tokenizer()

print("Building training prompts ...")
TRAINING_PAIRS = []
for tools, query, answers in PAIR_SPECS:
    msgs   = build_chat_messages(tools, query)
    prompt = _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    response = json.dumps(answers, ensure_ascii=False)
    TRAINING_PAIRS.append((prompt, response))

print(f"  {len(TRAINING_PAIRS)} training pairs ready.")

# Load base model
print(f"\nLoading base model ({MODEL_NAME}) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,  # fp16 for GTX 1070 (no bfloat16 on Pascal)
)

# If a saved best LoRA exists, load it as the starting point
if os.path.exists(LORA_BEST_DIR):
    print(f"Resuming from saved LoRA at {LORA_BEST_DIR} ...")
    model = PeftModel.from_pretrained(base_model, LORA_BEST_DIR, is_trainable=True)
else:
    print("Applying fresh LoRA adapters ...")
    lora_cfg = LoraConfig(
        task_type=TaskType.CAUSAL_LM,
        r=LORA_R,
        lora_alpha=LORA_ALPHA,
        lora_dropout=LORA_DROPOUT,
        target_modules=LORA_TARGETS,
        bias="none",
    )
    model = get_peft_model(base_model, lora_cfg)

model.print_trainable_parameters()

# ---------------------------------------------------------------------------
# Pre-training eval
# ---------------------------------------------------------------------------

print("\n--- Pre-training evaluation ---")
pre_results = evaluate_function_calling(model, _TOKENIZER, batch_size=EVAL_BATCH)
print(f"name_accuracy : {pre_results['name_accuracy']:.4f}  ({int(pre_results['name_accuracy'] * pre_results['n'])}/{pre_results['n']})")
print(f"parse_rate    : {pre_results['parse_rate']:.4f}")

# ---------------------------------------------------------------------------
# Fine-tuning loop (time-bounded)
# ---------------------------------------------------------------------------

print(f"\n--- Fine-tuning for up to {TIME_BUDGET}s ---")

model.train()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

t_start_training = time.time()
total_training_time = 0.0
step = 0
epoch = 0
completed_steps = 0

# Pre-tokenize all training pairs
batch_data = make_sft_batch(TRAINING_PAIRS, _TOKENIZER)

while True:
    epoch += 1
    if epoch > EPOCHS:
        break

    # Shuffle indices for this epoch
    indices = torch.randperm(len(TRAINING_PAIRS)).tolist()

    micro_losses = []
    grad_step = 0

    for batch_start in range(0, len(indices), MICRO_BATCH):
        torch.cuda.synchronize()
        t0 = time.time()

        idx = indices[batch_start:batch_start + MICRO_BATCH]

        input_ids = batch_data["input_ids"][idx].to(device)
        attention_mask = batch_data["attention_mask"][idx].to(device)
        labels    = batch_data["labels"][idx].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        micro_losses.append(loss.item() * GRAD_ACCUM)
        grad_step += 1

        if grad_step % GRAD_ACCUM == 0 or batch_start + MICRO_BATCH >= len(indices):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            optimizer.zero_grad()
            completed_steps += 1

        torch.cuda.synchronize()
        t1 = time.time()
        total_training_time += (t1 - t0)

        avg_loss = sum(micro_losses) / len(micro_losses) if micro_losses else 0.0
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\repoch {epoch} step {completed_steps} | loss: {avg_loss:.4f} | remaining: {remaining:.0f}s    ", end="", flush=True)

        step += 1

        if total_training_time >= TIME_BUDGET:
            break

    if total_training_time >= TIME_BUDGET:
        break

print()  # newline after \r

if completed_steps == 0:
    print("WARNING: no training steps completed — increase TIME_BUDGET or reduce data size.")

# ---------------------------------------------------------------------------
# Post-training eval
# ---------------------------------------------------------------------------

print("\n--- Post-training evaluation ---")
model.eval()
post_results = evaluate_function_calling(model, _TOKENIZER, batch_size=EVAL_BATCH)
print(f"name_accuracy : {post_results['name_accuracy']:.4f}  ({int(post_results['name_accuracy'] * post_results['n'])}/{post_results['n']})")
print(f"parse_rate    : {post_results['parse_rate']:.4f}")

delta = post_results["name_accuracy"] - pre_results["name_accuracy"]
print(f"\ndelta name_accuracy: {delta:+.4f}")

# ---------------------------------------------------------------------------
# Save LoRA if improved
# ---------------------------------------------------------------------------

if delta > 0:
    print(f"Improvement detected — saving LoRA adapter to {LORA_BEST_DIR}")
    os.makedirs(LORA_BEST_DIR, exist_ok=True)
    model.save_pretrained(LORA_BEST_DIR)
    _TOKENIZER.save_pretrained(LORA_BEST_DIR)
else:
    print("No improvement — not saving this run's weights.")

# ---------------------------------------------------------------------------
# Final summary (grep-friendly)
# ---------------------------------------------------------------------------

t_end = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"pre_accuracy:     {pre_results['name_accuracy']:.6f}")
print(f"post_accuracy:    {post_results['name_accuracy']:.6f}")
print(f"delta_accuracy:   {delta:+.6f}")
print(f"parse_rate:       {post_results['parse_rate']:.6f}")
print(f"training_seconds: {total_training_time:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"completed_steps:  {completed_steps}")
print(f"epochs:           {epoch}")
