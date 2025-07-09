from modules.planner import TaskPlanner
from modules.orchestrator import ToolOrchestrator
from modules.executor import ActionExecutor
from modules.reflection import AdaptiveReasoner
from llm_client import GeminiLLM
import json

# === Step 1: User input ===
user_query = (
    "Plan a trip to Singapore from 19 July 2025 to 24 July 2025 for 2 people. "
    "The estimated budget for the trip is 3000USD. Recommend attractions and create an itinerary, "
    "prioritizing activities from the following list: ['sightseeing', 'shopping', 'festivals']. "
    "Please also consider their preferred travel pace (e.g., fast-paced, relaxed) and accommodation style (e.g., luxury, boutique, budget) if provided."
)

# === Step 2: Use LLM to interpret user query ===
llm = GeminiLLM()
perception_prompt = f"""
Extract structured travel request information from the following user input:

---
{user_query}
---

Return a JSON object with keys: destination, start_date, end_date, num_people, budget, preferences, pace, accommodation_style.
Dates must be in ISO format (YYYY-MM-DD).
"""
response = llm.call(perception_prompt)
query_data = json.loads(response)  # LLM must return valid JSON

# === Step 3: Plan tasks ===
task_plan = TaskPlanner().decompose(query_data)

# === Step 4: Orchestrate tools ===
tool_outputs = ToolOrchestrator().execute_plan(task_plan, query_data)

# === Step 5: Reflect and adapt ===
final_plan = AdaptiveReasoner().reflect_and_adapt(tool_outputs, query_data)

# === Step 6: Execute final plan and output ===
ActionExecutor().deliver_final_plan(final_plan)