import time
import random
from collections import defaultdict

# --- Configuration and Simulation Environment ---

# Simulate LLM responses for travel planning
def simulate_llm_response(prompt, model_name="gemini-2.0-flash"):
    """
    Simulates an LLM response based on the input prompt.
    Adds a small delay to simulate API calls.
    Returns (response_text, simulated_tokens).
    """
    time.sleep(random.uniform(0.1, 0.3)) # Simulate network latency
    prompt_lower = prompt.lower()
    simulated_tokens = len(prompt) // 2 + random.randint(30, 80) # Estimate tokens

    # --- Travel-specific LLM responses ---
    if "plan:" in prompt_lower or "reason:" in prompt_lower or "thought:" in prompt_lower:
        if "estimated break-down costs" in prompt_lower or "budget" in prompt_lower:
            return "Thought: I need to calculate the estimated costs for travel. Action: get_travel_costs('Singapore', '3 days')", simulated_tokens + 100
        elif "places to eat" in prompt_lower or "restaurants" in prompt_lower:
            return "Thought: I should search for popular dining spots in Singapore. Action: search_restaurants('Singapore', 'local food')", simulated_tokens + 90
        elif "place to visit" in prompt_lower or "attractions" in prompt_lower:
            return "Thought: I'll look up top attractions in Singapore. Action: find_attractions('Singapore', 'must-see')", simulated_tokens + 95
        elif "places to stay" in prompt_lower or "hotels" in prompt_lower:
            return "Thought: I need to find suitable accommodation. Action: search_hotels('Singapore', 'mid-range')", simulated_tokens + 85
        elif "alternative dates" in prompt_lower or "re-plan" in prompt_lower:
            return "Thought: The initial plan had issues. I need to find alternative flight options or adjust the budget. Action: get_travel_costs('Singapore', '3 days', alternative=True)", simulated_tokens + 110
        elif "synthesize" in prompt_lower or "summarize" in prompt_lower or "final answer" in prompt_lower:
            return "Thought: I have gathered all necessary travel information. Action: present_summary()", simulated_tokens + 70
        else:
            return "Thought: I need to break down the travel request. Action: plan_travel('Singapore', '3 days')", simulated_tokens + 70
    
    # --- Simulated Tool Responses (LLM explaining tool output) ---
    elif "get_travel_costs(" in prompt_lower:
        if "too expensive" in prompt_lower:
            return "Response: I found a budget for $800. Is this acceptable?", simulated_tokens + 60
        return "Response: Estimated budget for 3 days in Singapore: Flights $150, Accommodation $200, Food $150, Activities $100. Total: $600.", simulated_tokens + 150
    elif "search_restaurants(" in prompt_lower:
        return "Response: Top places to eat: Lau Pa Sat (hawker), Maxwell Food Centre, Newton Food Centre.", simulated_tokens + 120
    elif "find_attractions(" in prompt_lower:
        return "Response: Must-visit attractions: Gardens by the Bay, Marina Bay Sands, Sentosa Island, National Gallery Singapore.", simulated_tokens + 130
    elif "search_hotels(" in prompt_lower:
        return "Response: Recommended hotels: Hotel G (mid-range), The Fullerton Hotel (luxury), Capsule Pod Boutique Hostel (budget).", simulated_tokens + 140
    elif "present_summary()" in prompt_lower:
        return "Final Answer: Your 3-day Singapore trip includes an estimated budget of $600, food at hawker centers, and visits to major attractions.", simulated_tokens + 160
    else:
        return "I am processing your travel request.", simulated_tokens + 50

# Simulate external travel-related tool calls
def simulate_tool_call(tool_name, tool_args):
    """
    Simulates an external tool call.
    Returns (tool_output, success_status).
    """
    time.sleep(random.uniform(0.05, 0.2)) # Simulate tool execution time
    
    # Simulate a potential failure for reflection purposes
    if "alternative=True" in str(tool_args):
        if tool_name == "get_travel_costs":
            return "Alternative flight found, total budget now $800 (was $950).", True
    
    if tool_name == "get_travel_costs":
        if random.random() < 0.2: # 20% chance of initial cost being 'too expensive'
            return "Initial cost estimate too expensive: $950. Please consider alternative dates or budget options.", False
        return "Estimated breakdown: Flights ~$150, Accommodation ~$200, Food ~$150, Activities ~$100.", True
    elif tool_name == "search_restaurants":
        return "Successfully retrieved restaurant list: [Lau Pa Sat, Maxwell, Newton Food Centre].", True
    elif tool_name == "find_attractions":
        return "Successfully retrieved attractions: [Gardens by the Bay, MBS, Sentosa].", True
    elif tool_name == "search_hotels":
        return "Successfully retrieved hotel options: [Hotel G, Fullerton, Capsule Pod].", True
    elif tool_name == "plan_travel" or tool_name == "present_summary":
        return "Operation successful.", True
    else:
        return f"Tool '{tool_name}' not recognized or failed.", False

# --- Agent Design Patterns Implementations ---

class Agent:
    def __init__(self, name):
        self.name = name
        self.total_tokens_used = 0
        self.steps_taken = 0
        self.memory = [] # Simple memory to store interaction history

    def _call_llm(self, prompt):
        response, tokens = simulate_llm_response(prompt)
        self.total_tokens_used += tokens
        self.memory.append(f"LLM Call: {prompt} -> {response}")
        return response

    def _call_tool(self, tool_name, tool_args):
        output, success = simulate_tool_call(tool_name, tool_args)
        self.memory.append(f"Tool Call: {tool_name}({tool_args}) -> {output} (Success: {success})")
        return output, success

    def reset_metrics(self):
        self.total_tokens_used = 0
        self.steps_taken = 0
        self.memory = []

    def execute(self, task_description):
        raise NotImplementedError("Each agent pattern must implement its own execute method.")

class ReactiveAgent(Agent):
    """
    A simple reactive agent that performs a single LLM call to get an action
    and then a single tool call based on that action. It doesn't plan or reflect.
    """
    def __init__(self):
        super().__init__("ReactiveAgent")

    def execute(self, task_description):
        self.reset_metrics()
        start_time = time.time()
        self.steps_taken = 1

        # Direct LLM call to get an "action"
        llm_response = self._call_llm(f"Given the travel task: '{task_description}', what direct action should I take and with what arguments? Format: Action: action(args)")
        
        # Simple parsing to extract tool and args (e.g., get_travel_costs('Singapore', '3 days'))
        try:
            action_part = llm_response.split("Action: ")[-1]
            tool_name = action_part.split("(")[0].strip()
            tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")
            
            tool_output, tool_success = self._call_tool(tool_name, tool_args)
            final_output = f"Reactive Agent completed: {tool_output}"
            task_success = tool_success
        except (IndexError, AttributeError):
            final_output = f"Reactive Agent failed to parse action from LLM response: {llm_response}"
            task_success = False

        end_time = time.time()
        latency = end_time - start_time
        return task_success, latency, self.total_tokens_used, self.steps_taken, final_output

class ReActAgent(Agent):
    """
    A ReAct (Reasoning and Acting) agent that interleaves thought (LLM) and action (Tool) steps.
    It attempts multiple steps to solve the task.
    """
    def __init__(self):
        super().__init__("ReActAgent")

    def execute(self, task_description, max_steps=4): # Increased max_steps for more complex tasks
        self.reset_metrics()
        start_time = time.time()
        current_state = task_description
        task_success = False
        final_output = "Task incomplete."

        for step in range(max_steps):
            self.steps_taken += 1
            # Step 1: LLM reasons and suggests an action
            prompt = f"Current travel planning state: '{current_state}'. What is your next thought and action? Format: Thought: ... Action: action(args)"
            llm_response = self._call_llm(prompt)

            if "Thought:" in llm_response and "Action:" in llm_response:
                thought = llm_response.split("Thought:")[1].split("Action:")[0].strip()
                action_part = llm_response.split("Action:")[1].strip()

                try:
                    tool_name = action_part.split("(")[0].strip()
                    tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")

                    # Step 2: Perform the action (tool call)
                    tool_output, tool_success = self._call_tool(tool_name, tool_args)
                    current_state = f"Thought: {thought}. Action taken: {action_part}. Observation: {tool_output}"

                    # Simplified success condition for travel planning breakdown
                    if tool_success and ("Estimated breakdown" in tool_output or "Successfully retrieved" in tool_output or "Final Answer" in tool_output):
                        final_output = tool_output
                        if "Final Answer" in tool_output: # ReAct can reach a summary
                            task_success = True
                            break
                        # Continue if more parts of the breakdown are needed (e.g., after getting costs, get restaurants)
                        # This would require more sophisticated LLM prompting for chaining tasks
                        # For this simulation, we'll aim for a "good enough" output for a single aspect
                        if tool_name == "get_travel_costs" and "Estimated breakdown" in tool_output:
                            current_state += f"\nNow get places to eat." # Simple chaining
                        elif tool_name == "search_restaurants" and "Successfully retrieved" in tool_output:
                             current_state += f"\nNow get places to visit."
                        elif tool_name == "find_attractions" and "Successfully retrieved" in tool_output:
                             current_state += f"\nNow get places to stay."
                        elif tool_name == "search_hotels" and "Successfully retrieved" in tool_output:
                            current_state += f"\nNow summarize the plan. Action: present_summary()" # Final step for summary
                        
                    elif "too expensive" in tool_output.lower() or "failed" in tool_output.lower():
                        current_state = f"Previous attempt failed: {tool_output}. What should I do next? Consider alternative options." # Prompt for re-evaluation
                except (IndexError, AttributeError):
                    final_output = f"ReAct Agent failed to parse action from LLM response: {llm_response}"
                    break
            else:
                final_output = f"ReAct Agent received unparseable LLM response: {llm_response}"
                break
        
        # Final check if task was successful after all steps
        if "Final Answer" in final_output:
            task_success = True

        end_time = time.time()
        latency = end_time - start_time
        return task_success, latency, self.total_tokens_used, self.steps_taken, final_output

class ReflectiveAgent(ReActAgent):
    """
    A Reflective Agent extends ReAct by having an explicit reflection step
    if the initial attempt or a tool call is deemed unsuccessful (e.g., cost too high).
    """
    def __init__(self):
        super().__init__()
        self.name = "ReflectiveAgent"

    def execute(self, task_description, max_initial_steps=3, max_reflection_retries=1):
        self.reset_metrics()
        start_time = time.time()
        current_state = task_description
        task_success = False
        final_output = "Task incomplete."
        
        # Initial ReAct execution
        for step in range(max_initial_steps):
            self.steps_taken += 1
            prompt = f"Current travel planning state: '{current_state}'. What is your next thought and action? Format: Thought: ... Action: action(args)"
            llm_response = self._call_llm(prompt)

            if "Thought:" in llm_response and "Action:" in llm_response:
                thought = llm_response.split("Thought:")[1].split("Action:")[0].strip()
                action_part = llm_response.split("Action:")[1].strip()

                try:
                    tool_name = action_part.split("(")[0].strip()
                    tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")

                    tool_output, tool_success = self._call_tool(tool_name, tool_args)
                    current_state = f"Thought: {thought}. Action taken: {action_part}. Observation: {tool_output}"

                    if tool_success and ("Estimated breakdown" in tool_output or "Successfully retrieved" in tool_output or "Final Answer" in tool_output):
                        final_output = tool_output
                        if "Final Answer" in tool_output:
                            task_success = True
                            break
                        
                        # Chaining logic, similar to ReAct, but specifically for this complex query
                        if tool_name == "get_travel_costs" and "Estimated breakdown" in tool_output:
                            current_state += f"\nNow get places to eat."
                        elif tool_name == "search_restaurants" and "Successfully retrieved" in tool_output:
                             current_state += f"\nNow get places to visit."
                        elif tool_name == "find_attractions" and "Successfully retrieved" in tool_output:
                             current_state += f"\nNow get places to stay."
                        elif tool_name == "search_hotels" and "Successfully retrieved" in tool_output:
                            current_state += f"\nNow summarize the plan. Action: present_summary()"

                    else:
                        # If tool call indicates an issue, prepare for reflection
                        current_state = f"Previous action ('{action_part}') resulted in: '{tool_output}'. This was not fully successful. Reflect on what went wrong and propose a new plan or action. "
                except (IndexError, AttributeError):
                    final_output = f"Reflective Agent failed to parse action in initial phase: {llm_response}"
                    break
            else:
                final_output = f"Reflective Agent received unparseable LLM response in initial phase: {llm_response}"
                break
        
        # Reflection phase if initial attempt wasn't fully successful and there's a reason to reflect
        if not task_success and "Reflect on what went wrong" in current_state:
            for retry in range(max_reflection_retries):
                self.steps_taken += 1
                # LLM reflects on the failure and suggests a new approach
                reflection_prompt = f"Problem: {current_state} What adjustments should be made to achieve the task? Think step-by-step. Action: new_action(args)"
                llm_reflection_response = self._call_llm(reflection_prompt)
                
                if "Action:" in llm_reflection_response:
                    action_part = llm_reflection_response.split("Action:")[-1].strip()
                    try:
                        tool_name = action_part.split("(")[0].strip()
                        tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")
                        tool_output, tool_success = self._call_tool(tool_name, tool_args)
                        
                        if tool_success and ("Estimated breakdown" in tool_output or "Successfully retrieved" in tool_output or "Final Answer" in tool_output):
                            final_output = f"Reflective Agent (after reflection): {tool_output}"
                            if "Final Answer" in tool_output:
                                task_success = True
                            break # Reflection succeeded, break out of retry loop
                        else:
                            current_state = f"Reflection attempt {retry+1} also resulted in: '{tool_output}'. Further adjustments needed."
                    except (IndexError, AttributeError):
                        final_output = f"Reflective Agent failed to parse action after reflection: {llm_reflection_response}"
                        break
                else:
                    final_output = f"Reflective Agent received unparseable LLM reflection: {llm_reflection_response}"
                    break

        # Final check if task was successful after all steps
        if "Final Answer" in final_output:
            task_success = True

        end_time = time.time()
        latency = end_time - start_time
        return task_success, latency, self.total_tokens_used, self.steps_taken, final_output


# --- Evaluation Runner ---

def evaluate_pattern(agent_instance, task, num_runs=5):
    """
    Evaluates a single agent pattern over multiple runs for a given task.
    Collects metrics and returns averages.
    """
    results = {
        "successes": 0,
        "latencies": [],
        "token_usages": [],
        "steps_taken": [],
        "last_outputs": []
    }

    for _ in range(num_runs):
        success, latency, tokens, steps, output = agent_instance.execute(task)
        if success:
            results["successes"] += 1
        results["latencies"].append(latency)
        results["token_usages"].append(tokens)
        results["steps_taken"].append(steps)
        results["last_outputs"].append(output) # Store for qualitative review

    avg_latency = sum(results["latencies"]) / num_runs
    avg_tokens = sum(results["token_usages"]) / num_runs
    avg_steps = sum(results["steps_taken"]) / num_runs
    success_rate = (results["successes"] / num_runs) * 100

    return {
        "success_rate": success_rate,
        "avg_latency": avg_latency,
        "avg_tokens": avg_tokens,
        "avg_steps": avg_steps,
        "all_outputs": results["last_outputs"] # Can be used for qualitative analysis
    }

# --- Main Execution for Comparison ---

if __name__ == "__main__":
    # Define complex travel planning query as the main task
    travel_query_task = "I want to travel to Singapore in next 3 days. Give me estimated break-down costs, places to eat, place to visit, places to stay, etc."

    agents = {
        "Reactive Agent": ReactiveAgent(),
        "ReAct Agent": ReActAgent(),
        "Reflective Agent": ReflectiveAgent()
    }

    comparison_results = defaultdict(dict)

    print("Starting evaluation of Agentic AI Design Patterns for Travel Planning...\n")

    for agent_name, agent_instance in agents.items():
        print(f"--- Evaluating {agent_name} ---")
        print(f"  Running for task: '{travel_query_task}'")
        # Evaluate each agent on the single, complex travel query
        metrics = evaluate_pattern(agent_instance, travel_query_task, num_runs=5)
        comparison_results[agent_name][travel_query_task] = metrics
        print(f"    Success Rate: {metrics['success_rate']:.2f}%")
        print(f"    Avg Latency: {metrics['avg_latency']:.4f} seconds")
        print(f"    Avg Tokens Used: {metrics['avg_tokens']:.2f}")
        print(f"    Avg Steps Taken: {metrics['avg_steps']:.2f}\n")

    print("\n--- Summary of Comparison Results for Travel Planning ---")
    for agent_name, agent_tasks in comparison_results.items():
        print(f"\n## {agent_name}")
        for task_description, metrics in agent_tasks.items():
            print(f"### Task: '{task_description}'")
            print(f"- Success Rate: {metrics['success_rate']:.2f}%")
            print(f"- Avg Latency: {metrics['avg_latency']:.4f} s")
            print(f"- Avg Tokens Used: {metrics['avg_tokens']:.2f}")
            print(f"- Avg Steps Taken: {metrics['avg_steps']:.2f}")
            print(f"- Sample Output (first run): {metrics['all_outputs'][0]}") # Show one sample output for context

    print("\nEvaluation complete. This simulation demonstrates a framework for comparing design patterns for travel agents.")
    print("For a real-world scientific comparison, replace simulations with actual API calls to LLMs and travel service APIs.")
    print("Consider more complex success criteria, human evaluation, and a larger variety of nuanced travel queries.")

