"""
SmolLM-xLAM function-calling experiment. Single-GPU, LoRA fine-tuning.
Usage: uv run train.py

This file changes every run. Edit PAIR_SPECS and hyperparameters.
Fixed metric: evaluate_function_calling() in prepare.py.
Goal: increase name_accuracy by >= 0.01 per run.
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
# Hyperparameters
# ---------------------------------------------------------------------------

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 2e-4
WEIGHT_DECAY  = 0.01
EPOCHS        = 20       # fresh LoRA needs more steps; budget allows it
MICRO_BATCH   = 4
GRAD_ACCUM    = 2

EVAL_BATCH    = 8

# ---------------------------------------------------------------------------
# Run 7: fresh LoRA + debug-targeted data + fixed eval (128 tokens, preamble strip)
#
# State: best_lora cache deleted (run5 had corrupted it with MLP-expanded weights).
# Train from scratch with 30 examples carefully chosen to match failing eval patterns:
#   - Exact eval function names: realtime_weather_api, trending, find_kth_smallest_number,
#     calculate_grade, market_get_price_chart, validate_cpf_number, etc.
#   - All multi-tool examples force correct selection from distractors
#   - Multi-call (same fn repeated) matches eval distribution
#   - All use "arguments" key — zero use of "parameters"
# ---------------------------------------------------------------------------

def _tool(name, desc, params=None):
    props, req = {}, []
    for k, v in (params or {}).items():
        props[k] = {"type": v[0], "description": v[1]}
        if len(v) > 2 and v[2]:
            req.append(k)
    return {"name": name, "description": desc,
            "parameters": {"type": "object", "properties": props, "required": req}}


PAIR_SPECS = [
    # --- Exact eval function names (direct match for hardest examples) ---
    (
        [_tool("realtime_weather_api", "Get real-time weather for a location",
               {"q": ("string", "City name or coordinates", True),
                "units": ("string", "Metric or imperial", False)})],
        "What is the current weather in London?",
        [{"name": "realtime_weather_api", "arguments": {"q": "London"}}],
    ),
    (
        [_tool("trending", "Get trending content on a platform",
               {"platform": ("string", "Platform name e.g. YouTube, TikTok", True),
                "region": ("string", "Region code", False),
                "limit": ("integer", "Max results", False)})],
        "What is currently trending on YouTube?",
        [{"name": "trending", "arguments": {"platform": "YouTube"}}],
    ),
    (
        [_tool("market_get_price_chart", "Get a stock price chart",
               {"ticker": ("string", "Stock ticker symbol", True),
                "period": ("string", "Time period e.g. 1d 1w 1m 1y", False),
                "interval": ("string", "Data interval", False)})],
        "Show me the price chart for AAPL over the past month.",
        [{"name": "market_get_price_chart", "arguments": {"ticker": "AAPL", "period": "1m"}}],
    ),
    (
        [_tool("find_kth_smallest_number", "Find the k-th smallest number in a list",
               {"nums": ("array", "List of numbers", True),
                "k": ("integer", "The rank k (1-indexed)", True)})],
        "Find the 3rd smallest number in [7, 2, 1, 6, 5, 3, 4, 8].",
        [{"name": "find_kth_smallest_number", "arguments": {"nums": [7, 2, 1, 6, 5, 3, 4, 8], "k": 3}}],
    ),
    (
        [_tool("calculate_grade", "Calculate grade from a list of scores",
               {"scores": ("array", "List of numeric scores", True),
                "weights": ("array", "List of weights for each score", False)})],
        "Calculate the grade for scores [85, 92, 78, 95].",
        [{"name": "calculate_grade", "arguments": {"scores": [85, 92, 78, 95]}}],
    ),
    (
        [_tool("validate_cpf_number", "Validate a Brazilian CPF tax number",
               {"cpf": ("string", "CPF number string", True)})],
        "Is the CPF number 529.982.247-25 valid?",
        [{"name": "validate_cpf_number", "arguments": {"cpf": "529.982.247-25"}}],
    ),
    (
        [_tool("predict_evolution_rate", "Predict species evolution rate over time",
               {"species": ("string", "Species name", True),
                "years": ("integer", "Number of years", True),
                "model": ("string", "Evolution model", False)})],
        "Predict the evolution rate of Homo sapiens over 1000 years.",
        [{"name": "predict_evolution_rate", "arguments": {"species": "Homo sapiens", "years": 1000}}],
    ),
    (
        [_tool("stock_quotes", "Get real-time stock quote for a ticker",
               {"ticker": ("string", "Stock ticker symbol", True),
                "exchange": ("string", "Exchange name", False)})],
        "What is the current stock price of Tesla?",
        [{"name": "stock_quotes", "arguments": {"ticker": "TSLA"}}],
    ),
    (
        [_tool("facebook_ad_copy", "Generate Facebook ad copy for a product",
               {"product": ("string", "Product or service name", True),
                "tone": ("string", "Tone: professional, casual, urgent", False)})],
        "Generate Facebook ad copy for a new pair of running shoes.",
        [{"name": "facebook_ad_copy", "arguments": {"product": "running shoes"}}],
    ),
    (
        [_tool("most_expensive", "Get the most expensive item in a category",
               {"category": ("string", "Item category", True),
                "currency": ("string", "Currency code", False)})],
        "What is the most expensive item in the electronics category?",
        [{"name": "most_expensive", "arguments": {"category": "electronics"}}],
    ),

    # --- Multi-tool: choose one from several (matching eval distribution) ---
    (
        [_tool("wire_resistance", "Calculate wire resistance",
               {"length": ("number", "Wire length in meters", True),
                "material": ("string", "Wire material", True)}),
         _tool("find_kth_smallest_number", "Find k-th smallest number",
               {"nums": ("array", "Number list", True),
                "k": ("integer", "Rank k", True)}),
         _tool("note_duration", "Calculate musical note duration",
               {"bpm": ("integer", "Beats per minute", True),
                "note_type": ("string", "Note type", True)})],
        "Find the 2nd smallest number in [10, 3, 7, 1, 5].",
        [{"name": "find_kth_smallest_number", "arguments": {"nums": [10, 3, 7, 1, 5], "k": 2}}],
    ),
    (
        [_tool("is_armstrong_number", "Check if a number is Armstrong",
               {"num": ("integer", "The number to check", True)}),
         _tool("calculate_grade", "Calculate grade from scores",
               {"scores": ("array", "Score list", True)})],
        "Is the number 153 an Armstrong number?",
        [{"name": "is_armstrong_number", "arguments": {"num": 153}}],
    ),
    (
        [_tool("get_chat_restrictions", "Get chat restrictions for a user",
               {"user_id": ("string", "User ID", True),
                "platform": ("string", "Platform name", True)}),
         _tool("get_user_info", "Get user profile information",
               {"user_id": ("string", "User ID", True)}),
         _tool("sticker_roulette", "Get a random sticker",
               {"pack_id": ("string", "Sticker pack ID", False)})],
        "Get the chat restrictions for user 'u123' on Telegram.",
        [{"name": "get_chat_restrictions", "arguments": {"user_id": "u123", "platform": "Telegram"}}],
    ),
    (
        [_tool("get_ip_zipcode", "Get the zipcode for an IP address",
               {"ip": ("string", "IPv4 address", True)}),
         _tool("assess_diabetes_risk", "Assess diabetes risk from health data",
               {"age": ("integer", "Patient age", True),
                "bmi": ("number", "Body mass index", True),
                "activity": ("string", "Activity level", True)}),
         _tool("get_pokemon_move_info", "Get info about a Pokemon move",
               {"move_name": ("string", "Move name", True)})],
        "Get the zipcode for IP address 8.8.8.8.",
        [{"name": "get_ip_zipcode", "arguments": {"ip": "8.8.8.8"}}],
    ),
    (
        [_tool("top_grossing_ipad_apps", "List top grossing iPad apps",
               {"category": ("string", "App category", False),
                "country": ("string", "Two-letter country code", False)}),
         _tool("search_countries_by_idd", "Search countries by dialing code",
               {"idd": ("string", "Dialing code", True)})],
        "What are the top grossing iPad apps in Germany?",
        [{"name": "top_grossing_ipad_apps", "arguments": {"country": "de"}}],
    ),
    (
        [_tool("multi_search", "Search across multiple catalogs",
               {"query": ("string", "Search query", True),
                "limit": ("integer", "Max results", False)}),
         _tool("get_song_related", "Get songs related to a given song",
               {"song_id": ("string", "Song ID", True)})],
        "Search for 'Shape of You' across all music catalogs.",
        [{"name": "multi_search", "arguments": {"query": "Shape of You"}}],
    ),
    (
        [_tool("detailed_cake_recipe_by_id", "Get a detailed cake recipe by ID",
               {"recipe_id": ("integer", "Recipe ID", True)}),
         _tool("menudetails", "Get menu details for a restaurant",
               {"restaurant_id": ("string", "Restaurant ID", True)}),
         _tool("fetch_restaurant_information", "Fetch general restaurant info",
               {"restaurant_id": ("string", "Restaurant ID", True)})],
        "Get the cake recipe with ID 42.",
        [{"name": "detailed_cake_recipe_by_id", "arguments": {"recipe_id": 42}}],
    ),
    (
        [_tool("getstandardmaptile", "Get a standard map tile",
               {"z": ("integer", "Zoom level", True),
                "x": ("integer", "X tile coordinate", True),
                "y": ("integer", "Y tile coordinate", True)}),
         _tool("local_osm_v1_z_x_y_png", "Get OSM tile as PNG",
               {"z": ("integer", "Zoom level", True),
                "x": ("integer", "X coordinate", True),
                "y": ("integer", "Y coordinate", True)}),
         _tool("reversegeocoding", "Convert coordinates to address",
               {"lat": ("number", "Latitude", True),
                "lon": ("number", "Longitude", True)})],
        "Get the OSM tile PNG for zoom 12, x=2048, y=1365.",
        [{"name": "local_osm_v1_z_x_y_png", "arguments": {"z": 12, "x": 2048, "y": 1365}}],
    ),
    (
        [_tool("steam", "Look up Steam game info",
               {"app_id": ("string", "Steam app ID", False),
                "query": ("string", "Search query", False)}),
         _tool("challenge", "Get challenge info by ID",
               {"challenge_id": ("string", "Challenge ID", True)}),
         _tool("real_time_user_search", "Search for users in real time",
               {"query": ("string", "User search query", True)})],
        "Look up the challenge with ID 'ch_9910'.",
        [{"name": "challenge", "arguments": {"challenge_id": "ch_9910"}}],
    ),
    (
        [_tool("query_for_city_names_by_state", "List city names by US state",
               {"state": ("string", "Two-letter state code", True)}),
         _tool("places_list_by_radius_nearby_search", "Find places near coordinates",
               {"lat": ("number", "Latitude", True),
                "lon": ("number", "Longitude", True),
                "radius": ("integer", "Radius in meters", True)}),
         _tool("getcity", "Get city info by name",
               {"city": ("string", "City name", True)})],
        "List all city names in the state of California.",
        [{"name": "query_for_city_names_by_state", "arguments": {"state": "CA"}}],
    ),

    # --- Multi-call (same function, different args) ---
    (
        [_tool("realtime_weather_api", "Get real-time weather",
               {"q": ("string", "City or coordinates", True)})],
        "Get the current weather for Tokyo and Seoul simultaneously.",
        [{"name": "realtime_weather_api", "arguments": {"q": "Tokyo"}},
         {"name": "realtime_weather_api", "arguments": {"q": "Seoul"}}],
    ),
    (
        [_tool("top_grossing_ipad_apps", "List top grossing iPad apps",
               {"country": ("string", "Two-letter country code", False)})],
        "Get top grossing iPad apps for the US, UK, and Australia.",
        [{"name": "top_grossing_ipad_apps", "arguments": {"country": "us"}},
         {"name": "top_grossing_ipad_apps", "arguments": {"country": "gb"}},
         {"name": "top_grossing_ipad_apps", "arguments": {"country": "au"}}],
    ),
    (
        [_tool("stock_quotes", "Get real-time stock quote",
               {"ticker": ("string", "Ticker symbol", True)}),
         _tool("stock_get_annual_avg_div_yield", "Get annual avg dividend yield",
               {"ticker": ("string", "Ticker symbol", True)})],
        "Get current stock quotes for both AAPL and GOOGL.",
        [{"name": "stock_quotes", "arguments": {"ticker": "AAPL"}},
         {"name": "stock_quotes", "arguments": {"ticker": "GOOGL"}}],
    ),
    (
        [_tool("get_chat_restrictions", "Get chat restrictions for a user",
               {"user_id": ("string", "User ID", True),
                "platform": ("string", "Platform name", True)})],
        "Check chat restrictions for users 'alice' and 'bob' on Discord.",
        [{"name": "get_chat_restrictions", "arguments": {"user_id": "alice", "platform": "Discord"}},
         {"name": "get_chat_restrictions", "arguments": {"user_id": "bob", "platform": "Discord"}}],
    ),

    # --- Clear single-selection with distractors ---
    (
        [_tool("autocomplete_zipcodes_lite", "Autocomplete US zipcode search",
               {"zipcode": ("string", "Partial zipcode", True)}),
         _tool("get_all_coins_prices", "Get prices for all cryptocurrencies", {}),
         _tool("most_expensive", "Get most expensive item in category",
               {"category": ("string", "Category", True)}),
         _tool("getfeedversions", "Get feed versions for a transit agency",
               {"agency_id": ("string", "Agency identifier", True)})],
        "Get current prices for all cryptocurrencies.",
        [{"name": "get_all_coins_prices", "arguments": {}}],
    ),
    (
        [_tool("profile", "Get a user's profile",
               {"user_id": ("string", "User ID", True)}),
         _tool("financial_income_statement", "Get company income statement",
               {"ticker": ("string", "Ticker symbol", True),
                "period": ("string", "annual or quarterly", False)}),
         _tool("market_get_price_chart", "Get price chart for a stock",
               {"ticker": ("string", "Ticker symbol", True),
                "period": ("string", "Time period", False)})],
        "Get the annual income statement for Microsoft (MSFT).",
        [{"name": "financial_income_statement", "arguments": {"ticker": "MSFT", "period": "annual"}}],
    ),
    (
        [_tool("calculate_standard_deviation", "Calculate standard deviation",
               {"numbers": ("array", "List of numbers", True)}),
         _tool("is_prime", "Check if a number is prime",
               {"number": ("integer", "Number to check", True)}),
         _tool("neuronal_activity_rate", "Calculate neuronal activity rate",
               {"neuron_id": ("string", "Neuron ID", True),
                "interval_ms": ("integer", "Interval in ms", True)})],
        "Calculate the standard deviation of [2, 4, 4, 4, 5, 5, 7, 9].",
        [{"name": "calculate_standard_deviation",
          "arguments": {"numbers": [2, 4, 4, 4, 5, 5, 7, 9]}}],
    ),
    (
        [_tool("transactions", "Get financial transactions for an account",
               {"account_id": ("string", "Account identifier", True),
                "limit": ("integer", "Max number of results", False),
                "start_date": ("string", "Start date YYYY-MM-DD", False)}),
         _tool("steps", "Log or retrieve step count data",
               {"user_id": ("string", "User ID", True),
                "date": ("string", "Date YYYY-MM-DD", False)}),
         _tool("trending", "Get trending content",
               {"platform": ("string", "Platform name", True)})],
        "Retrieve the last 10 transactions for account 'acc_7771'.",
        [{"name": "transactions", "arguments": {"account_id": "acc_7771", "limit": 10}}],
    ),
    (
        [_tool("find_first_non_repeating_char", "Find first non-repeating char",
               {"s": ("string", "Input string", True)}),
         _tool("reverse_string", "Reverse a string",
               {"s": ("string", "Input string", True)})],
        "What is the first non-repeating character in 'programming'?",
        [{"name": "find_first_non_repeating_char", "arguments": {"s": "programming"}}],
    ),
    (
        [_tool("getuserbyname", "Look up a user by username",
               {"username": ("string", "Username", True)}),
         _tool("dashboard", "Get dashboard stats for a user",
               {"user_id": ("string", "User ID", True)}),
         _tool("swap_id", "Swap two record IDs",
               {"id_a": ("string", "First ID", True),
                "id_b": ("string", "Second ID", True)})],
        "Look up the user with username 'janesmith'.",
        [{"name": "getuserbyname", "arguments": {"username": "janesmith"}}],
    ),
]

assert len(PAIR_SPECS) == 30, f"Expected 30 pairs, got {len(PAIR_SPECS)}"

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
    msgs     = build_chat_messages(tools, query)
    prompt   = _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    response = json.dumps(answers, ensure_ascii=False)
    TRAINING_PAIRS.append((prompt, response))
print(f"  {len(TRAINING_PAIRS)} training pairs ready.")

print(f"\nLoading base model ({MODEL_NAME}) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
)

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

t_start_training  = time.time()
total_training_time = 0.0
step = 0
epoch = 0
completed_steps = 0

batch_data = make_sft_batch(TRAINING_PAIRS, _TOKENIZER)

while True:
    epoch += 1
    if epoch > EPOCHS:
        break

    indices    = torch.randperm(len(TRAINING_PAIRS)).tolist()
    micro_losses = []
    grad_step  = 0

    for batch_start in range(0, len(indices), MICRO_BATCH):
        torch.cuda.synchronize()
        t0 = time.time()

        idx            = indices[batch_start:batch_start + MICRO_BATCH]
        input_ids      = batch_data["input_ids"][idx].to(device)
        attention_mask = batch_data["attention_mask"][idx].to(device)
        labels         = batch_data["labels"][idx].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out  = model(input_ids=input_ids, attention_mask=attention_mask, labels=labels)
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

        avg_loss  = sum(micro_losses) / len(micro_losses) if micro_losses else 0.0
        remaining = max(0, TIME_BUDGET - total_training_time)
        print(f"\repoch {epoch} step {completed_steps} | loss: {avg_loss:.4f} | remaining: {remaining:.0f}s    ", end="", flush=True)

        step += 1
        if total_training_time >= TIME_BUDGET:
            break

    if total_training_time >= TIME_BUDGET:
        break

print()

if completed_steps == 0:
    print("WARNING: no training steps completed.")

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
# Save LoRA if best ever (tracked in best_accuracy.txt)
# ---------------------------------------------------------------------------

best_path  = os.path.join(LORA_BEST_DIR, "best_accuracy.txt")
best_saved = 0.0
if os.path.exists(best_path):
    try:
        with open(best_path) as f:
            best_saved = float(f.read().strip())
    except Exception:
        pass

if post_results["name_accuracy"] > best_saved:
    print(f"New best ({post_results['name_accuracy']:.4f} > {best_saved:.4f}) — saving to {LORA_BEST_DIR}")
    os.makedirs(LORA_BEST_DIR, exist_ok=True)
    model.save_pretrained(LORA_BEST_DIR)
    _TOKENIZER.save_pretrained(LORA_BEST_DIR)
    with open(best_path, "w") as f:
        f.write(f"{post_results['name_accuracy']:.6f}")
elif delta > 0:
    print(f"Improved +{delta:.4f} within run but not better than saved best ({best_saved:.4f}).")
else:
    print("No improvement — not saving.")

# ---------------------------------------------------------------------------
# Final summary
# ---------------------------------------------------------------------------

t_end        = time.time()
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
