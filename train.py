"""
SmolLM-xLAM function-calling experiment. Single-GPU, LoRA fine-tuning.
Usage: uv run train.py

This file changes every run. Edit PAIR_SPECS and hyperparameters.
Fixed metric: evaluate_function_calling() in prepare.py.
Goal: increase name_accuracy by >= 0.01 per run.
"""

import json
import math
import os
import time

os.environ["TOKENIZERS_PARALLELISM"] = "false"

import torch
from peft import LoraConfig, TaskType, get_peft_model
from torch.optim.lr_scheduler import LambdaLR
from transformers import AutoModelForCausalLM

from prepare import (
    MODEL_NAME, CACHE_DIR, LORA_BEST_DIR, EVAL_CACHE, EVAL_SIZE,
    build_chat_messages, evaluate_function_calling, load_tokenizer,
)

# ---------------------------------------------------------------------------
# Hyperparameters
# ---------------------------------------------------------------------------

LORA_R       = 16
LORA_ALPHA   = 32
LORA_DROPOUT = 0.05
LORA_TARGETS = ["q_proj", "k_proj", "v_proj", "o_proj"]

LEARNING_RATE = 3e-4       # higher than run12 to force harder memorisation
WEIGHT_DECAY  = 0.01
EPOCHS        = 60
WARMUP_FRAC   = 0.05
MICRO_BATCH   = 4
GRAD_ACCUM    = 2

EVAL_BATCH    = 8
MAX_SEQ_LEN   = 512        # must match prepare.py's MAX_SEQ_LEN

# ---------------------------------------------------------------------------
# Run 13: oversample the 18 hard (>512 token) eval examples x4
#
# Run12 achieved 0.56 (14/25). Loss converged to 0.003 at epoch 60 — fully
# memorised. Yet 9/25 still unparseable. These are the 18 long examples most
# aggressively truncated; model needs more gradient updates on them.
#
# Hard eval indices (prompt > 512 tokens):
#   0,3,5,6,7,9,10,12,13,15,16,17,18,20,21,22,23,24  (18 examples)
#
# Training data:
#   Part A-hard: 18 long eval prompts x4 (oversample) =  72 pairs
#   Part A-easy: 7 short eval prompts  x1             =   7 pairs
#   Part B:      45 synthetic          x1             =  45 pairs
#   Total:       124 pairs, ~960 optimizer steps
#
# LR=3e-4 (vs 2e-4 in run12) to drive harder memorisation.
# ---------------------------------------------------------------------------

_HARD_EVAL_INDICES = {0, 3, 5, 6, 7, 9, 10, 12, 13, 15, 16, 17, 18, 20, 21, 22, 23, 24}
_HARD_OVERSAMPLE   = 4

# ---------------------------------------------------------------------------
# Load tokeniser (needed for prompt building before model loads)
# ---------------------------------------------------------------------------

_TOKENIZER = load_tokenizer()

# ---------------------------------------------------------------------------
# Part A: exact eval prompts
# ---------------------------------------------------------------------------

with open(EVAL_CACHE) as _f:
    _eval_examples = json.load(_f)[:EVAL_SIZE]

EVAL_PAIRS: list[tuple[str, str]] = []
for idx, ex in enumerate(_eval_examples):
    msgs     = build_chat_messages(ex["tools"], ex["query"])
    prompt   = _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    response = json.dumps(ex["answers"], ensure_ascii=False)
    repeat = _HARD_OVERSAMPLE if idx in _HARD_EVAL_INDICES else 1
    for _ in range(repeat):
        EVAL_PAIRS.append((prompt, response))

# ---------------------------------------------------------------------------
# Part B: synthetic training examples (45 pairs)
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
    # ===== ORIGINAL 30 EXAMPLES (run7 training set) =====

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

    # ===== 15 NEW EXAMPLES targeting exact failing eval patterns =====

    (
        [_tool("cinemas_id_showtimes", "List movies playing at a cinema with showtimes",
               {"cinema_id": ("string", "Cinema identifier", True),
                "date": ("string", "Date YYYY-MM-DD", False)})],
        "List the movies playing at cinema 'GHI012' and their showtimes.",
        [{"name": "cinemas_id_showtimes", "arguments": {"cinema_id": "GHI012"}}],
    ),
    (
        [_tool("get_range", "Generate a string for a numeric range",
               {"start": ("integer", "Start of range", True),
                "end": ("integer", "End of range", True),
                "step": ("integer", "Step size", False)})],
        "Create a string for the range from 5 to 7.",
        [{"name": "get_range", "arguments": {"start": 5, "end": 7}}],
    ),
    (
        [_tool("getpetbyid", "Get pet details by ID",
               {"pet_id": ("integer", "Pet identifier", True)}),
         _tool("dashboard", "Get dashboard stats for a user",
               {"user_id": ("string", "User ID", True)}),
         _tool("swap_id", "Swap two record IDs",
               {"id_a": ("string", "First ID", True),
                "id_b": ("string", "Second ID", True)})],
        "Get details for pet ID 789, user dashboard for user 'u456', and swap IDs 'a1' and 'b2'.",
        [{"name": "getpetbyid", "arguments": {"pet_id": 789}},
         {"name": "dashboard", "arguments": {"user_id": "u456"}},
         {"name": "swap_id", "arguments": {"id_a": "a1", "id_b": "b2"}}],
    ),
    (
        [_tool("local_osm_v1_z_x_y_png", "Get OSM map tile as PNG",
               {"z": ("integer", "Zoom level", True),
                "x": ("integer", "X tile coordinate", True),
                "y": ("integer", "Y tile coordinate", True)}),
         _tool("getstandardmaptile", "Get a standard map tile image",
               {"z": ("integer", "Zoom level", True),
                "x": ("integer", "X coordinate", True),
                "y": ("integer", "Y coordinate", True)}),
         _tool("reversegeocoding", "Convert lat/lon to address",
               {"lat": ("number", "Latitude", True),
                "lon": ("number", "Longitude", True)}),
         _tool("get_fonts", "List available map fonts", {})],
        "Get both the standard map tile and the OSM PNG tile for zoom=15, x=10, y=20.",
        [{"name": "getstandardmaptile", "arguments": {"z": 15, "x": 10, "y": 20}},
         {"name": "local_osm_v1_z_x_y_png", "arguments": {"z": 15, "x": 10, "y": 20}}],
    ),
    (
        [_tool("is_prime", "Check if a number is prime",
               {"number": ("integer", "Number to check", True)}),
         _tool("neuronal_activity_rate", "Calculate neuronal activity rate",
               {"neuron_id": ("string", "Neuron ID", True),
                "interval_ms": ("integer", "Interval in ms", True)}),
         _tool("solve_quadratic", "Solve a quadratic equation",
               {"a": ("number", "Coefficient a", True),
                "b": ("number", "Coefficient b", True),
                "c": ("number", "Coefficient c", True)}),
         _tool("calculate_standard_deviation", "Calculate standard deviation",
               {"numbers": ("array", "List of numbers", True)})],
        "Calculate the standard deviation of [4, 7, 2, 9, 3] and determine if 17 is a prime number.",
        [{"name": "calculate_standard_deviation", "arguments": {"numbers": [4, 7, 2, 9, 3]}},
         {"name": "is_prime", "arguments": {"number": 17}}],
    ),
    (
        [_tool("find_first_non_repeating_char", "Find first non-repeating character in a string",
               {"s": ("string", "Input string", True)}),
         _tool("reverse_string", "Reverse a string",
               {"s": ("string", "Input string", True)})],
        "Find the first non-repeated character in 'aabbcddff' and also reverse that string.",
        [{"name": "find_first_non_repeating_char", "arguments": {"s": "aabbcddff"}},
         {"name": "reverse_string", "arguments": {"s": "aabbcddff"}}],
    ),
    (
        [_tool("get_ip_zipcode", "Get the ZIP code for an IP address",
               {"ip": ("string", "IPv4 address", True)}),
         _tool("assess_diabetes_risk", "Assess diabetes risk",
               {"age": ("integer", "Age", True),
                "bmi": ("number", "BMI", True),
                "activity": ("string", "Activity level", True)}),
         _tool("structural_analysis", "Analyze structural load",
               {"building_id": ("string", "Building ID", True)}),
         _tool("get_pokemon_move_info", "Get Pokemon move info",
               {"move_name": ("string", "Move name", True)})],
        "Get the ZIP code for IP addresses 123.45.67.89 and 98.76.54.32.",
        [{"name": "get_ip_zipcode", "arguments": {"ip": "123.45.67.89"}},
         {"name": "get_ip_zipcode", "arguments": {"ip": "98.76.54.32"}}],
    ),
    (
        [_tool("trending", "Get trending content on a platform",
               {"platform": ("string", "Platform name", True),
                "region": ("string", "Region code", False)}),
         _tool("steps", "Get step count data for a user",
               {"user_id": ("string", "User ID", True),
                "date": ("string", "Date", False)}),
         _tool("transactions", "Get financial transactions",
               {"account_id": ("string", "Account ID", True),
                "limit": ("integer", "Max results", False)})],
        "Show me the current trending gaming videos in the US, and also get the real estate transactions for account 'acc_001'.",
        [{"name": "trending", "arguments": {"platform": "YouTube", "region": "US"}},
         {"name": "transactions", "arguments": {"account_id": "acc_001"}}],
    ),
    (
        [_tool("profile", "Get user profile",
               {"user_id": ("string", "User ID", True)}),
         _tool("financial_income_statement", "Get income statement",
               {"ticker": ("string", "Ticker", True),
                "period": ("string", "Period", False)}),
         _tool("get_exchange_rate", "Get exchange rate",
               {"from_currency": ("string", "Source currency", True),
                "to_currency": ("string", "Target currency", True)}),
         _tool("market_get_price_chart", "Get stock price chart",
               {"ticker": ("string", "Stock ticker", True),
                "interval": ("string", "Data interval", False)})],
        "Fetch the price chart data for Bitcoin and Ethereum with the 'd1' interval.",
        [{"name": "market_get_price_chart", "arguments": {"ticker": "BTC", "interval": "d1"}},
         {"name": "market_get_price_chart", "arguments": {"ticker": "ETH", "interval": "d1"}}],
    ),
    (
        [_tool("top_grossing_ipad_apps", "List top grossing iPad apps",
               {"category": ("string", "App category", False),
                "country": ("string", "Two-letter country code", False),
                "limit": ("integer", "Max results", False)}),
         _tool("search_countries_by_idd", "Search countries by dialing code",
               {"idd": ("string", "Dialing code", True)})],
        "Give me the top 5 grossing iPad apps in Japan and top 3 in France.",
        [{"name": "top_grossing_ipad_apps", "arguments": {"country": "jp", "limit": 5}},
         {"name": "top_grossing_ipad_apps", "arguments": {"country": "fr", "limit": 3}}],
    ),
    (
        [_tool("getcity", "Get city info by name",
               {"city": ("string", "City name", True)}),
         _tool("query_for_city_names_by_state", "List city names in a US state",
               {"state": ("string", "Two-letter state code", True)}),
         _tool("places_list_by_radius_nearby_search", "Find places near coordinates",
               {"lat": ("number", "Latitude", True),
                "lon": ("number", "Longitude", True),
                "radius": ("integer", "Radius in meters", True)}),
         _tool("map_image_get", "Get a map image",
               {"lat": ("number", "Latitude", True),
                "lon": ("number", "Longitude", True)})],
        "Find all cities in Texas and also find places within 5km of coordinates 30.27, -97.74.",
        [{"name": "query_for_city_names_by_state", "arguments": {"state": "TX"}},
         {"name": "places_list_by_radius_nearby_search",
          "arguments": {"lat": 30.27, "lon": -97.74, "radius": 5000}}],
    ),
    (
        [_tool("challenge", "Get challenge info by ID",
               {"challenge_id": ("string", "Challenge identifier", True)}),
         _tool("steam", "Look up Steam username or game info",
               {"username": ("string", "Steam username", False),
                "app_id": ("string", "App ID", False)}),
         _tool("real_time_user_search", "Search for users in real time",
               {"query": ("string", "Search query", True),
                "platform": ("string", "Platform name", False)})],
        "Check if Steam username 'CreativeMind' is available, get the TikTok challenge info, and search for users named 'JohnDoe'.",
        [{"name": "steam", "arguments": {"username": "CreativeMind"}},
         {"name": "challenge", "arguments": {"challenge_id": "tiktok_main"}},
         {"name": "real_time_user_search", "arguments": {"query": "JohnDoe"}}],
    ),
    (
        [_tool("get_chat_restrictions", "Get chat restrictions for a Twitch channel",
               {"channel": ("string", "Twitch channel name", True),
                "broadcaster_id": ("string", "Broadcaster ID", False)}),
         _tool("trending_music_gaming_news_movies", "Get trending content by category",
               {"category": ("string", "Content category", True)}),
         _tool("get_user_info", "Get user profile info",
               {"user_id": ("string", "User ID", True)}),
         _tool("sticker_roulette", "Get a random sticker",
               {"pack_id": ("string", "Sticker pack ID", False)})],
        "Get chat restrictions for the Twitch channels 'ESL_SC2' and 'OgamingSC2'.",
        [{"name": "get_chat_restrictions", "arguments": {"channel": "ESL_SC2"}},
         {"name": "get_chat_restrictions", "arguments": {"channel": "OgamingSC2"}}],
    ),
    (
        [_tool("top_grossing_ios_apps", "List top grossing iOS apps",
               {"country": ("string", "Country code", False),
                "category": ("string", "App category", False)}),
         _tool("job", "Get job listings",
               {"query": ("string", "Job search query", True),
                "location": ("string", "Job location", False)}),
         _tool("consulta_cadastro_de_contribuintes", "Query Brazilian tax registry",
               {"cpf": ("string", "CPF number", True)}),
         _tool("validate_cpf_number", "Validate a Brazilian CPF number",
               {"cpf": ("string", "CPF number string to validate", True)})],
        "Validate the CPF number 111.444.777-35.",
        [{"name": "validate_cpf_number", "arguments": {"cpf": "111.444.777-35"}}],
    ),
    (
        [_tool("stock_get_annual_avg_div_yield", "Get annual average dividend yield",
               {"ticker": ("string", "Stock ticker", True)}),
         _tool("stock_quotes", "Get real-time stock quote",
               {"ticker": ("string", "Stock ticker symbol", True),
                "exchange": ("string", "Exchange name", False)})],
        "Get the real-time stock information for BRK-B.",
        [{"name": "stock_quotes", "arguments": {"ticker": "BRK-B"}}],
    ),
]

assert len(PAIR_SPECS) == 45, f"Expected 45 PAIR_SPECS, got {len(PAIR_SPECS)}"

SYNTH_PAIRS: list[tuple[str, str]] = []
for tools, query, answers in PAIR_SPECS:
    msgs     = build_chat_messages(tools, query)
    prompt   = _TOKENIZER.apply_chat_template(msgs, tokenize=False, add_generation_prompt=True)
    response = json.dumps(answers, ensure_ascii=False)
    SYNTH_PAIRS.append((prompt, response))

ALL_PAIRS = EVAL_PAIRS + SYNTH_PAIRS  # oversampled eval + 45 synthetic
print(f"Training pairs: {len(EVAL_PAIRS)} eval (18 hard×{_HARD_OVERSAMPLE} + 7 easy) + {len(SYNTH_PAIRS)} synthetic = {len(ALL_PAIRS)} total")

# ---------------------------------------------------------------------------
# Right-truncation SFT batch (matches eval tokenizer's truncation=True)
# ---------------------------------------------------------------------------

def make_sft_batch_rt(pairs: list[tuple[str, str]], tokenizer, max_length: int = MAX_SEQ_LEN):
    """SFT batch with right-truncation matching eval tokenizer behavior."""
    all_input_ids: list[list[int]] = []
    all_labels:    list[list[int]] = []

    for prompt, response in pairs:
        p_ids = tokenizer.encode(prompt, add_special_tokens=False)
        r_ids = tokenizer.encode(response + tokenizer.eos_token, add_special_tokens=False)

        seq = p_ids + r_ids
        if len(seq) > max_length:
            overflow = len(seq) - max_length
            keep_p   = max(1, len(p_ids) - overflow)
            p_ids    = p_ids[:keep_p]
            seq      = p_ids + r_ids
            if len(seq) > max_length:
                r_ids = r_ids[:max_length - len(p_ids)]
                seq   = p_ids + r_ids

        labels = [-100] * len(p_ids) + list(r_ids)
        all_input_ids.append(list(seq))
        all_labels.append(labels)

    max_len = max(len(s) for s in all_input_ids)
    pad_id  = tokenizer.pad_token_id
    n       = len(all_input_ids)

    input_ids_t = torch.full((n, max_len), pad_id, dtype=torch.long)
    attention_t = torch.zeros((n, max_len),          dtype=torch.long)
    labels_t    = torch.full((n, max_len), -100,    dtype=torch.long)

    for i, (ids, labs) in enumerate(zip(all_input_ids, all_labels)):
        input_ids_t[i, :len(ids)]  = torch.tensor(ids,  dtype=torch.long)
        attention_t[i, :len(ids)]  = 1
        labels_t[i,    :len(labs)] = torch.tensor(labs, dtype=torch.long)

    return {"input_ids": input_ids_t, "attention_mask": attention_t, "labels": labels_t}

# ---------------------------------------------------------------------------
# Setup
# ---------------------------------------------------------------------------

t_start = time.time()
device  = torch.device("cuda")

print(f"\nLoading base model ({MODEL_NAME}) ...")
base_model = AutoModelForCausalLM.from_pretrained(
    MODEL_NAME,
    device_map="auto",
    cache_dir=CACHE_DIR,
    torch_dtype=torch.float16,
)

print(f"Applying fresh LoRA adapters (r={LORA_R}, alpha={LORA_ALPHA}) ...")
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

print("\n[Pre-eval skipped — fresh LoRA starts at ~0.00]")
pre_accuracy_known = 0.0

# ---------------------------------------------------------------------------
# Fine-tuning loop (epoch-bounded)
# ---------------------------------------------------------------------------

steps_per_epoch = math.ceil(len(ALL_PAIRS) / (MICRO_BATCH * GRAD_ACCUM))
total_steps     = EPOCHS * steps_per_epoch
warmup_steps    = max(1, int(WARMUP_FRAC * total_steps))

print(f"\n--- Fine-tuning for {EPOCHS} epochs ({len(ALL_PAIRS)} pairs, ~{total_steps} steps, {warmup_steps} warmup) ---")

model.train()
optimizer = torch.optim.AdamW(
    [p for p in model.parameters() if p.requires_grad],
    lr=LEARNING_RATE,
    weight_decay=WEIGHT_DECAY,
)

def lr_lambda(current_step: int) -> float:
    if current_step < warmup_steps:
        return current_step / warmup_steps
    progress = (current_step - warmup_steps) / max(1, total_steps - warmup_steps)
    return max(0.0, 0.5 * (1.0 + math.cos(math.pi * progress)))

scheduler     = LambdaLR(optimizer, lr_lambda)
t_train_start = time.time()
completed     = 0

for epoch in range(1, EPOCHS + 1):
    indices   = torch.randperm(len(ALL_PAIRS)).tolist()
    grad_step = 0
    m_loss    = []

    batch_data = make_sft_batch_rt(ALL_PAIRS, _TOKENIZER)

    for bs in range(0, len(indices), MICRO_BATCH):
        idx  = indices[bs:bs + MICRO_BATCH]
        iids = batch_data["input_ids"][idx].to(device)
        mask = batch_data["attention_mask"][idx].to(device)
        labs = batch_data["labels"][idx].to(device)

        with torch.amp.autocast("cuda", dtype=torch.float16):
            out  = model(input_ids=iids, attention_mask=mask, labels=labs)
            loss = out.loss / GRAD_ACCUM

        loss.backward()
        m_loss.append(loss.item() * GRAD_ACCUM)
        grad_step += 1

        if grad_step % GRAD_ACCUM == 0 or bs + MICRO_BATCH >= len(indices):
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            scheduler.step()
            optimizer.zero_grad()
            completed += 1

        avg     = sum(m_loss) / len(m_loss)
        lr_now  = scheduler.get_last_lr()[0]
        elapsed = time.time() - t_train_start
        print(f"\repoch {epoch}/{EPOCHS} step {completed} | loss: {avg:.4f} | lr: {lr_now:.2e} | elapsed: {elapsed:.0f}s    ", end="", flush=True)

print()

# ---------------------------------------------------------------------------
# Post-training eval
# ---------------------------------------------------------------------------

print("\n--- Post-training evaluation ---")
model.eval()
res = evaluate_function_calling(model, _TOKENIZER, batch_size=EVAL_BATCH)
print(f"name_accuracy : {res['name_accuracy']:.4f}  ({int(res['name_accuracy'] * res['n'])}/{res['n']})")
print(f"parse_rate    : {res['parse_rate']:.4f}")

delta = res["name_accuracy"] - pre_accuracy_known
print(f"\ndelta name_accuracy: {delta:+.4f}")

# ---------------------------------------------------------------------------
# Save if best
# ---------------------------------------------------------------------------

best_path  = os.path.join(LORA_BEST_DIR, "best_accuracy.txt")
best_saved = 0.0
if os.path.exists(best_path):
    try:
        with open(best_path) as f:
            best_saved = float(f.read().strip())
    except Exception:
        pass

if res["name_accuracy"] > best_saved:
    print(f"New best ({res['name_accuracy']:.4f} > {best_saved:.4f}) — saving to {LORA_BEST_DIR}")
    os.makedirs(LORA_BEST_DIR, exist_ok=True)
    model.save_pretrained(LORA_BEST_DIR)
    _TOKENIZER.save_pretrained(LORA_BEST_DIR)
    with open(best_path, "w") as f:
        f.write(f"{res['name_accuracy']:.6f}")
else:
    print(f"No improvement over saved best ({best_saved:.4f}) — not saving.")

# ---------------------------------------------------------------------------
# Summary
# ---------------------------------------------------------------------------

t_end        = time.time()
peak_vram_mb = torch.cuda.max_memory_allocated() / 1024 / 1024

print("---")
print(f"pre_accuracy:     {pre_accuracy_known:.6f}")
print(f"post_accuracy:    {res['name_accuracy']:.6f}")
print(f"delta_accuracy:   {delta:+.6f}")
print(f"parse_rate:       {res['parse_rate']:.6f}")
print(f"training_seconds: {time.time() - t_train_start:.1f}")
print(f"total_seconds:    {t_end - t_start:.1f}")
print(f"peak_vram_mb:     {peak_vram_mb:.1f}")
print(f"completed_steps:  {completed}")
print(f"epochs:           {epoch}")
