"""
Benchmark Logger
=================
Logs every LLM call in the system to a SQLite database for
cost tracking and savings analysis.

This is the proof layer -- it records exactly how much each
request costs through the router vs. what it would have cost
if every call used Claude Opus at full price.

Database: benchmarks/router.db
Export:   benchmarks/results.json
"""

from __future__ import annotations

import json
import os
import sqlite3
import sys
from datetime import datetime, timezone

# ── Make project root importable when run directly ───────────
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

from src.config.models import ModelRole, MODELS


# ── Paths ────────────────────────────────────────────────────
_PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
_DB_DIR = os.path.join(_PROJECT_ROOT, "benchmarks")
_DB_PATH = os.path.join(_DB_DIR, "router.db")
_JSON_PATH = os.path.join(_DB_DIR, "results.json")

# ── Opus baseline pricing (USD per 1K tokens) ───────────────
# Claude Opus 4 pricing as of 2025:
#   Input:  $15.00 per 1M tokens  = $0.015 per 1K tokens
#   Output: $75.00 per 1M tokens  = $0.075 per 1K tokens
_OPUS_COST_PER_1K_INPUT: float = 0.015
_OPUS_COST_PER_1K_OUTPUT: float = 0.075


# ── Database initialization ─────────────────────────────────

def init_db() -> None:
    """Create the SQLite database and tables if they don't exist.

    Creates two tables:
      - llm_calls: one row per LLM API call
      - requests:  one row per user request (aggregated costs)

    The database file is created at benchmarks/router.db.
    """
    os.makedirs(_DB_DIR, exist_ok=True)

    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS llm_calls (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id      TEXT NOT NULL,
            step_name       TEXT NOT NULL,
            provider        TEXT NOT NULL,
            model_id        TEXT NOT NULL,
            input_tokens    INTEGER NOT NULL,
            output_tokens   INTEGER NOT NULL,
            cost_usd        REAL NOT NULL,
            timestamp       TEXT NOT NULL
        )
    """)

    cursor.execute("""
        CREATE TABLE IF NOT EXISTS requests (
            id                      INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id              TEXT UNIQUE NOT NULL,
            user_prompt_preview     TEXT,
            complexity              TEXT,
            planner_used            TEXT,
            total_cost_usd          REAL,
            opus_equivalent_cost_usd REAL,
            savings_percent         REAL,
            verification_choice     TEXT,
            timestamp               TEXT NOT NULL
        )
    """)

    conn.commit()
    conn.close()


# ── Cost calculation helper ──────────────────────────────────

def _calculate_cost(model_id: str, input_tokens: int, output_tokens: int) -> float:
    """Calculate the cost of an LLM call based on config pricing.

    Looks up the model in the MODELS config to find per-token costs.
    If the model isn't found (e.g. external model), returns 0.0.

    Args:
        model_id:      The model identifier string.
        input_tokens:  Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        Cost in USD.
    """
    for config in MODELS.values():
        if config.model_id == model_id:
            input_cost = (input_tokens / 1000) * config.cost_per_1k_input
            output_cost = (output_tokens / 1000) * config.cost_per_1k_output
            return round(input_cost + output_cost, 8)

    # Model not in config -- return 0 (free-tier / unknown)
    return 0.0


def _calculate_opus_cost(input_tokens: int, output_tokens: int) -> float:
    """Calculate what the same tokens would cost at Opus pricing.

    This is the baseline for savings comparison.

    Args:
        input_tokens:  Total input tokens across all calls.
        output_tokens: Total output tokens across all calls.

    Returns:
        Equivalent cost in USD if Opus were used for everything.
    """
    input_cost = (input_tokens / 1000) * _OPUS_COST_PER_1K_INPUT
    output_cost = (output_tokens / 1000) * _OPUS_COST_PER_1K_OUTPUT
    return round(input_cost + output_cost, 8)


# ── Logging functions ────────────────────────────────────────

def log_llm_call(
    request_id: str,
    step_name: str,
    provider: str,
    model_id: str,
    input_tokens: int,
    output_tokens: int,
) -> float:
    """Log a single LLM API call to the database.

    Calculates the cost from config/models.py pricing and inserts
    a row into the llm_calls table.

    Args:
        request_id:    Groups all calls for one user request.
        step_name:     Which pipeline step made this call
                       (e.g. "classifier", "layer1", "layer2",
                       "opus_planner", "cheap_planner", "executor").
        provider:      Provider name (e.g. "groq", "gemini").
        model_id:      The model identifier string.
        input_tokens:  Number of input tokens used.
        output_tokens: Number of output tokens generated.

    Returns:
        The calculated cost in USD.
    """
    cost = _calculate_cost(model_id, input_tokens, output_tokens)
    timestamp = datetime.now(timezone.utc).isoformat()

    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT INTO llm_calls
            (request_id, step_name, provider, model_id,
             input_tokens, output_tokens, cost_usd, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?)
    """, (request_id, step_name, provider, model_id,
          input_tokens, output_tokens, cost, timestamp))

    conn.commit()
    conn.close()

    return cost


def log_request(
    request_id: str,
    user_prompt: str,
    complexity: str,
    planner_used: str,
    verification_choice: str,
    all_calls: list[dict],
) -> dict:
    """Log a complete user request with aggregated costs.

    Sums all individual call costs to get total_cost_usd,
    calculates what the same tokens would cost at Opus pricing,
    and computes the savings percentage.

    Args:
        request_id:          Unique identifier for this request.
        user_prompt:         The user's original prompt.
        complexity:          Classified complexity level string.
        planner_used:        Which planner was used ("opus" or "cheap").
        verification_choice: User's verification choice string.
        all_calls:           List of dicts, each with keys:
                             input_tokens, output_tokens, cost_usd.

    Returns:
        A summary dict with total cost, opus equivalent, and savings.
    """
    # ── Sum actual costs ─────────────────────────────────────
    total_cost = sum(call.get("cost_usd", 0.0) for call in all_calls)
    total_input_tokens = sum(call.get("input_tokens", 0) for call in all_calls)
    total_output_tokens = sum(call.get("output_tokens", 0) for call in all_calls)

    # ── Calculate Opus equivalent ────────────────────────────
    opus_equivalent = _calculate_opus_cost(total_input_tokens, total_output_tokens)

    # ── Calculate savings ────────────────────────────────────
    if opus_equivalent > 0:
        savings_percent = round((1 - total_cost / opus_equivalent) * 100, 2)
    else:
        savings_percent = 0.0

    timestamp = datetime.now(timezone.utc).isoformat()
    prompt_preview = user_prompt[:100]

    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        INSERT OR REPLACE INTO requests
            (request_id, user_prompt_preview, complexity, planner_used,
             total_cost_usd, opus_equivalent_cost_usd, savings_percent,
             verification_choice, timestamp)
        VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?)
    """, (request_id, prompt_preview, complexity, planner_used,
          total_cost, opus_equivalent, savings_percent,
          verification_choice, timestamp))

    conn.commit()
    conn.close()

    return {
        "request_id": request_id,
        "total_cost_usd": round(total_cost, 8),
        "opus_equivalent_cost_usd": round(opus_equivalent, 8),
        "savings_percent": savings_percent,
        "total_input_tokens": total_input_tokens,
        "total_output_tokens": total_output_tokens,
    }


# ── Query functions ──────────────────────────────────────────

def get_savings_summary() -> dict:
    """Query the requests table and return an aggregate summary.

    Returns:
        A dict with keys:
            total_requests          (int)   - number of logged requests.
            average_savings_percent (float) - mean savings across all requests.
            total_cost_usd          (float) - total actual cost.
            total_opus_equivalent_usd (float) - total Opus baseline cost.
            total_saved_usd         (float) - total USD saved.
    """
    conn = sqlite3.connect(_DB_PATH)
    cursor = conn.cursor()

    cursor.execute("""
        SELECT
            COUNT(*) as total_requests,
            COALESCE(AVG(savings_percent), 0) as avg_savings,
            COALESCE(SUM(total_cost_usd), 0) as total_cost,
            COALESCE(SUM(opus_equivalent_cost_usd), 0) as total_opus,
            COALESCE(SUM(opus_equivalent_cost_usd) - SUM(total_cost_usd), 0) as total_saved
        FROM requests
    """)

    row = cursor.fetchone()
    conn.close()

    return {
        "total_requests": row[0],
        "average_savings_percent": round(row[1], 2),
        "total_cost_usd": round(row[2], 8),
        "total_opus_equivalent_usd": round(row[3], 8),
        "total_saved_usd": round(row[4], 8),
    }


def export_to_json() -> None:
    """Export all database data to benchmarks/results.json.

    Creates a JSON file with two top-level keys:
      - llm_calls: list of all individual LLM call records
      - requests:  list of all request summaries
      - summary:   aggregate savings data

    This is the file you present as proof of cost savings.
    """
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()

    # ── Export llm_calls ─────────────────────────────────────
    cursor.execute("SELECT * FROM llm_calls ORDER BY id")
    llm_calls = [dict(row) for row in cursor.fetchall()]

    # ── Export requests ──────────────────────────────────────
    cursor.execute("SELECT * FROM requests ORDER BY id")
    requests = [dict(row) for row in cursor.fetchall()]

    conn.close()

    # ── Build summary ────────────────────────────────────────
    summary = get_savings_summary()

    export_data = {
        "exported_at": datetime.now(timezone.utc).isoformat(),
        "summary": summary,
        "requests": requests,
        "llm_calls": llm_calls,
    }

    with open(_JSON_PATH, "w", encoding="utf-8") as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"  [Logger] Exported to {_JSON_PATH}")


# ── Inline test ──────────────────────────────────────────────
if __name__ == "__main__":
    import uuid

    print("=" * 70)
    print("  Benchmark Logger - Test")
    print("=" * 70)

    # ── Initialize the database ──────────────────────────────
    print("\n  [Test 1] Initializing database...")
    init_db()
    print(f"  Database created at: {_DB_PATH}")

    # ── Simulate a full request with multiple LLM calls ──────
    req_id = str(uuid.uuid4())[:8]
    print(f"\n  [Test 2] Logging fake calls for request: {req_id}")

    fake_calls = [
        {
            "step_name": "classifier",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "input_tokens": 150,
            "output_tokens": 10,
        },
        {
            "step_name": "layer1_optimizer",
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "input_tokens": 800,
            "output_tokens": 500,
        },
        {
            "step_name": "layer2_evaluator",
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "input_tokens": 1200,
            "output_tokens": 200,
        },
        {
            "step_name": "cheap_planner",
            "provider": "groq",
            "model_id": "openai/gpt-oss-120b",
            "input_tokens": 600,
            "output_tokens": 1000,
        },
        {
            "step_name": "executor_step_1",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "input_tokens": 500,
            "output_tokens": 800,
        },
        {
            "step_name": "executor_step_2",
            "provider": "gemini",
            "model_id": "gemini-2.5-flash",
            "input_tokens": 900,
            "output_tokens": 600,
        },
    ]

    all_logged_calls: list[dict] = []

    for call in fake_calls:
        cost = log_llm_call(
            request_id=req_id,
            step_name=call["step_name"],
            provider=call["provider"],
            model_id=call["model_id"],
            input_tokens=call["input_tokens"],
            output_tokens=call["output_tokens"],
        )
        call["cost_usd"] = cost
        all_logged_calls.append(call)
        print(f"    {call['step_name']:25s} | {call['model_id']:25s} | ${cost:.6f}")

    # ── Log the full request ─────────────────────────────────
    print(f"\n  [Test 3] Logging request summary...")
    summary = log_request(
        request_id=req_id,
        user_prompt="Build a REST API for a todo app with user authentication and PostgreSQL",
        complexity="medium",
        planner_used="cheap",
        verification_choice="accept_optimized",
        all_calls=all_logged_calls,
    )

    print(f"    Total cost (actual):  ${summary['total_cost_usd']:.6f}")
    print(f"    Total cost (Opus):    ${summary['opus_equivalent_cost_usd']:.6f}")
    print(f"    Savings:              {summary['savings_percent']}%")
    print(f"    Input tokens total:   {summary['total_input_tokens']}")
    print(f"    Output tokens total:  {summary['total_output_tokens']}")

    # ── Get aggregate summary ────────────────────────────────
    print(f"\n  [Test 4] Aggregate savings summary:")
    agg = get_savings_summary()
    print(f"    Total requests:       {agg['total_requests']}")
    print(f"    Average savings:      {agg['average_savings_percent']}%")
    print(f"    Total actual cost:    ${agg['total_cost_usd']:.6f}")
    print(f"    Total Opus equiv:     ${agg['total_opus_equivalent_usd']:.6f}")
    print(f"    Total saved:          ${agg['total_saved_usd']:.6f}")

    # ── Export to JSON ───────────────────────────────────────
    print(f"\n  [Test 5] Exporting to JSON...")
    export_to_json()

    print(f"\n{'=' * 70}")
    print("  Done.")
    print(f"{'=' * 70}\n")
