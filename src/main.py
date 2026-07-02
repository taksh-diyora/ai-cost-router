"""
AI Cost Router - Main Entry Point
=================================
Connects all pipeline components into a single execution flow:
missing info -> classify -> optimize -> evaluate -> verify -> plan -> execute
"""

import sys
import uuid
import sqlite3
import os
from dotenv import load_dotenv

from src.pipeline.missing_info_detector import run_missing_info_loop
from src.pipeline.classifier import hybrid_classify, ComplexityLevel
from src.pipeline.iteration_loop import run_optimization_loop
from src.pipeline.user_verification import handle_verification
from src.planner import get_plan
from src.pipeline.executor import execute_direct, execute_plan
from src.benchmark.logger import init_db, log_request, get_savings_summary, export_to_json, _DB_PATH

def safe_print(text: str) -> None:
    """Print text safely in Windows consoles to avoid UnicodeEncodeError."""
    print(text.encode("ascii", errors="replace").decode("ascii"))

def get_calls_for_request(request_id: str) -> list[dict]:
    """Fetch all logged LLM calls for a specific request ID."""
    if not os.path.exists(_DB_PATH):
        return []
    conn = sqlite3.connect(_DB_PATH)
    conn.row_factory = sqlite3.Row
    cursor = conn.cursor()
    cursor.execute("SELECT input_tokens, output_tokens, cost_usd FROM llm_calls WHERE request_id = ?", (request_id,))
    rows = cursor.fetchall()
    conn.close()
    return [dict(row) for row in rows]

def run_pipeline(user_prompt: str) -> dict:
    """Execute the full AI Cost Router pipeline for a user prompt."""
    load_dotenv()
    request_id = uuid.uuid4().hex[:8]
    init_db()
    
    print(f"\n[{request_id}] Starting Request")
    
    # 1. Missing Info Loop
    enriched_prompt = run_missing_info_loop(user_prompt, request_id=request_id)
    
    # 2. Classification
    classification_result = hybrid_classify(enriched_prompt, request_id=request_id)
    complexity = classification_result["complexity"]
    
    # Default ambiguous complexity to MEDIUM
    if complexity is None:
        print(f"[{request_id}] Classification ambiguous, defaulting to MEDIUM.")
        complexity = ComplexityLevel.MEDIUM
        
    print(f"[{request_id}] Classification: {complexity.name}")
    
    # 3. Direct Execution (LOW complexity)
    if complexity == ComplexityLevel.LOW:
        print(f"[{request_id}] Routing to Direct Execution (LOW complexity)")
        execution_result = execute_direct(enriched_prompt, request_id=request_id)
        
        all_calls = get_calls_for_request(request_id)
        summary = log_request(
            request_id=request_id,
            user_prompt=user_prompt,
            complexity="LOW",
            planner_used="none",
            verification_choice="none",
            all_calls=all_calls
        )
        
        print("\n=== FINAL OUTPUT ===")
        safe_print(execution_result["final_output"])
        print("====================")
        print(f"Savings: {summary['savings_percent']}% | Cost: ${summary['total_cost_usd']:.6f}")
        
        export_to_json()
        return execution_result
        
    # 4. Optimization & Planning (MEDIUM or HIGH complexity)
    while True:
        # a. Optimization Loop
        loop_result = run_optimization_loop(enriched_prompt, complexity, request_id=request_id)
        
        # b. User Verification
        verification_result = handle_verification(enriched_prompt, loop_result)
        
        # c. Restart Check
        if verification_result["restart"]:
            print(f"\n[{request_id}] Restarting pipeline with modified prompt...")
            enriched_prompt = verification_result["prompt_to_use"]
            classification_result = hybrid_classify(enriched_prompt, request_id=request_id)
            complexity = classification_result["complexity"]
            if complexity is None:
                complexity = ComplexityLevel.MEDIUM
            continue
        else:
            break
            
    # d. Planning
    plan_result = get_plan(
        prompt=verification_result["prompt_to_use"],
        complexity=complexity,
        task_type=loop_result["task_type"],
        user_choice=verification_result["choice"],
        request_id=request_id
    )
    
    # e. Execution
    execution_result = execute_plan(
        plan=plan_result["plan"],
        original_prompt=enriched_prompt,
        task_type=loop_result["task_type"],
        request_id=request_id
    )
    
    # f. Logging
    all_calls = get_calls_for_request(request_id)
    summary = log_request(
        request_id=request_id,
        user_prompt=user_prompt,
        complexity=complexity.name,
        planner_used=plan_result["planner_used"],
        verification_choice=verification_result["choice"].name,
        all_calls=all_calls
    )
    
    # g. Print final savings summary
    print("\n" + "=" * 50)
    print("               SAVINGS SUMMARY")
    print("=" * 50)
    print(f"  Request ID:       {summary['request_id']}")
    print(f"  Actual Cost:      ${summary['total_cost_usd']:.6f}")
    print(f"  Opus Cost:        ${summary['opus_equivalent_cost_usd']:.6f}")
    print(f"  Savings:          {summary['savings_percent']}%")
    print("=" * 50 + "\n")
    
    export_to_json()
    return execution_result

if __name__ == "__main__":
    print("================================")
    print("       AI Cost Router           ")
    print("================================")
    print("Enter your prompt (type END on a new line to finish):")
    lines = []
    while True:
        try:
            line = input()
            if line.strip() == "END":
                break
            lines.append(line)
        except EOFError:
            break
            
    user_input = "\n".join(lines).strip()
    if user_input:
        run_pipeline(user_input)
