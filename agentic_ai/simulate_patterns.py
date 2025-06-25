import time
import random
from collections import defaultdict

# --- Configuration and Simulation Environment ---

# Simulate LLM responses based on keywords in the prompt
def simulate_llm_response(prompt, model_name="gemini-2.0-flash"):
    """
    Simulates an LLM response based on the input prompt.
    Adds a small delay to simulate API calls.
    Returns (response_text, simulated_tokens).
    """
    time.sleep(random.uniform(0.1, 0.3)) # Simulate network latency
    prompt_lower = prompt.lower()
    simulated_tokens = len(prompt) // 2 + random.randint(10, 50) # Estimate tokens

    if "plan:" in prompt_lower or "reason:" in prompt_lower or "thought:" in prompt_lower:
        if "research science paper" in prompt_lower:
            return "Thought: I need to search for keywords related to 'quantum computing' and 'future applications'. Then I will synthesize the findings. Action: search('quantum computing future applications')", simulated_tokens + 80
        elif "fix code error" in prompt_lower:
            return "Thought: I need to analyze the error message, identify the problematic line, and suggest a correction. Action: analyze_code('error details')", simulated_tokens + 70
        elif "analyze market trends" in prompt_lower:
            return "Thought: I should use a financial data tool to get recent trends for 'AI stocks'. Action: get_financial_data('AI stocks')", simulated_tokens + 75
        else:
            return "Thought: I need to break down the request and determine the best action. Action: default_action()", simulated_tokens + 60
    elif "search(" in prompt_lower:
        return "Found 10 relevant papers. Key insight: Quantum machine learning is an emerging field.", simulated_tokens + 120
    elif "analyze_code(" in prompt_lower:
        return "Error identified in line 23: 'IndexError: list index out of range'. Suggestion: Check list bounds.", simulated_tokens + 90
    elif "get_financial_data(" in prompt_lower:
        return "AI stock trends: Strong growth in Q1, slight dip in Q2, projected recovery in Q3.", simulated_tokens + 100
    elif "synthesize" in prompt_lower or "summarize" in prompt_lower:
        return "The future of quantum computing in machine learning shows great promise, particularly in optimization problems.", simulated_tokens + 95
    elif "final answer:" in prompt_lower or "output:" in prompt_lower:
        return "The agent successfully completed the task.", simulated_tokens + 50
    else:
        return "I am processing your request.", simulated_tokens + 40

# Simulate an external tool call (e.g., an API)
def simulate_tool_call(tool_name, tool_args):
    """
    Simulates an external tool call.
    Returns (tool_output, success_status).
    """
    time.sleep(random.uniform(0.05, 0.2)) # Simulate tool execution time

    if tool_name == "search":
        if "quantum computing" in tool_args:
            return "Search results for 'quantum computing': [Paper A, Paper B, Article C].", True
        else:
            return "No relevant search results found.", False
    elif tool_name == "analyze_code":
        if "IndexError" in tool_args:
            return "Code analysis: Line 23 error confirmed. Fix needed.", False # Indicate failure for reflection pattern
        else:
            return "Code analysis: No major issues found.", True
    elif tool_name == "get_financial_data":
        if "AI stocks" in tool_args:
            return "Financial data for AI stocks: Q1 +15%, Q2 -5%, Q3 projection +10%.", True
        else:
            return "Failed to retrieve financial data for given query.", False
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
    A simple reactive agent that performs a single LLM call and then a single tool call.
    It doesn't plan or reflect.
    """
    def __init__(self):
        super().__init__("ReactiveAgent")

    def execute(self, task_description):
        self.reset_metrics()
        start_time = time.time()
        self.steps_taken = 1

        # Direct LLM call to get an "action"
        llm_response = self._call_llm(f"Given the task: '{task_description}', what direct action should I take and with what arguments? Format: action(args)")
        
        # Simple parsing to extract tool and args (e.g., search('query'))
        try:
            action_part = llm_response.split("Action: ")[-1]
            tool_name = action_part.split("(")[0]
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
    """
    def __init__(self):
        super().__init__("ReActAgent")

    def execute(self, task_description, max_steps=3):
        self.reset_metrics()
        start_time = time.time()
        current_state = task_description
        task_success = False
        final_output = "Task incomplete."

        for step in range(max_steps):
            self.steps_taken += 1
            # Step 1: LLM reasons and suggests an action
            prompt = f"Current task: '{current_state}'. What is your next thought and action? Format: Thought: ... Action: action(args)"
            llm_response = self._call_llm(prompt)

            if "Thought:" in llm_response and "Action:" in llm_response:
                thought = llm_response.split("Thought:")[1].split("Action:")[0].strip()
                action_part = llm_response.split("Action:")[1].strip()

                try:
                    tool_name = action_part.split("(")[0]
                    tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")

                    # Step 2: Perform the action (tool call)
                    tool_output, tool_success = self._call_tool(tool_name, tool_args)
                    current_state = f"Thought: {thought}. Action taken: {action_part}. Observation: {tool_output}"

                    if tool_success and "Key insight:" in tool_output or "No major issues found" in tool_output or "Strong growth" in tool_output: # Simplified success condition
                        final_output = tool_output
                        task_success = True
                        break # Task considered successful
                    elif "fix needed" in tool_output.lower() or "failed" in tool_output.lower():
                        current_state = f"Previous attempt failed. {tool_output}. What should I do next?"
                except (IndexError, AttributeError):
                    final_output = f"ReAct Agent failed to parse action from LLM response: {llm_response}"
                    break
            else:
                final_output = f"ReAct Agent received unparseable LLM response: {llm_response}"
                break

        end_time = time.time()
        latency = end_time - start_time
        return task_success, latency, self.total_tokens_used, self.steps_taken, final_output

class ReflectiveAgent(ReActAgent):
    """
    A Reflective Agent extends ReAct by having an explicit reflection step
    if the initial attempt is deemed unsuccessful.
    """
    def __init__(self):
        super().__init__()
        self.name = "ReflectiveAgent"

    def execute(self, task_description, max_initial_steps=2, max_reflection_retries=1):
        self.reset_metrics()
        start_time = time.time()
        current_state = task_description
        task_success = False
        final_output = "Task incomplete."
        
        # Initial ReAct execution
        for step in range(max_initial_steps):
            self.steps_taken += 1
            prompt = f"Current task: '{current_state}'. What is your next thought and action? Format: Thought: ... Action: action(args)"
            llm_response = self._call_llm(prompt)

            if "Thought:" in llm_response and "Action:" in llm_response:
                thought = llm_response.split("Thought:")[1].split("Action:")[0].strip()
                action_part = llm_response.split("Action:")[1].strip()

                try:
                    tool_name = action_part.split("(")[0]
                    tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")

                    tool_output, tool_success = self._call_tool(tool_name, tool_args)
                    current_state = f"Thought: {thought}. Action taken: {action_part}. Observation: {tool_output}"

                    if tool_success and ("Key insight:" in tool_output or "No major issues found" in tool_output or "Strong growth" in tool_output):
                        final_output = tool_output
                        task_success = True
                        break # Task considered successful
                    else:
                        # If tool call fails or indicates an issue, prepare for reflection
                        current_state = f"Previous action ('{action_part}') resulted in: '{tool_output}'. This was not fully successful. Reflect on what went wrong and propose a new plan or action. "
                except (IndexError, AttributeError):
                    final_output = f"Reflective Agent failed to parse action in initial phase: {llm_response}"
                    break
            else:
                final_output = f"Reflective Agent received unparseable LLM response in initial phase: {llm_response}"
                break
        
        # Reflection phase if initial attempt wasn't fully successful
        if not task_success and "Reflect on what went wrong" in current_state:
            for retry in range(max_reflection_retries):
                self.steps_taken += 1
                # LLM reflects on the failure and suggests a new approach
                reflection_prompt = f"Problem: {current_state} What adjustments should be made to achieve the task? Think step-by-step. Action: new_action(args)"
                llm_reflection_response = self._call_llm(reflection_prompt)
                
                if "Action:" in llm_reflection_response:
                    action_part = llm_reflection_response.split("Action:")[-1].strip()
                    try:
                        tool_name = action_part.split("(")[0]
                        tool_args = action_part.split("(")[1].split(")")[0].strip("'\"")
                        tool_output, tool_success = self._call_tool(tool_name, tool_args)
                        
                        if tool_success and ("Key insight:" in tool_output or "No major issues found" in tool_output or "Strong growth" in tool_output):
                            final_output = f"Reflective Agent (after reflection): {tool_output}"
                            task_success = True
                            break
                        else:
                            current_state = f"Reflection attempt {retry+1} also resulted in: '{tool_output}'. Further adjustments needed."
                    except (IndexError, AttributeError):
                        final_output = f"Reflective Agent failed to parse action after reflection: {llm_reflection_response}"
                        break
                else:
                    final_output = f"Reflective Agent received unparseable LLM reflection: {llm_reflection_response}"
                    break


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
    tasks = {
        "Task 1: Research the future applications of quantum computing.": "quantum computing future applications",
        "Task 2: Fix the 'IndexError' in the provided code snippet.": "IndexError: list index out of range", # Simulate an error that needs fixing
        "Task 3: Analyze recent market trends for AI stocks.": "AI stocks"
    }

    agents = {
        "Reactive Agent": ReactiveAgent(),
        "ReAct Agent": ReActAgent(),
        "Reflective Agent": ReflectiveAgent()
    }

    comparison_results = defaultdict(dict)

    print("Starting evaluation of Agentic AI Design Patterns...\n")

    for agent_name, agent_instance in agents.items():
        print(f"--- Evaluating {agent_name} ---")
        for task_description, _ in tasks.items(): # Use task_description for the agent
            print(f"  Running for task: '{task_description}'")
            metrics = evaluate_pattern(agent_instance, task_description, num_runs=5)
            comparison_results[agent_name][task_description] = metrics
            print(f"    Success Rate: {metrics['success_rate']:.2f}%")
            print(f"    Avg Latency: {metrics['avg_latency']:.4f} seconds")
            print(f"    Avg Tokens Used: {metrics['avg_tokens']:.2f}")
            print(f"    Avg Steps Taken: {metrics['avg_steps']:.2f}\n")

    print("\n--- Summary of Comparison Results ---")
    for agent_name, agent_tasks in comparison_results.items():
        print(f"\n## {agent_name}")
        for task_description, metrics in agent_tasks.items():
            print(f"### Task: '{task_description}'")
            print(f"- Success Rate: {metrics['success_rate']:.2f}%")
            print(f"- Avg Latency: {metrics['avg_latency']:.4f} s")
            print(f"- Avg Tokens Used: {metrics['avg_tokens']:.2f}")
            print(f"- Avg Steps Taken: {metrics['avg_steps']:.2f}")
            # print(f"- Sample Outputs: {metrics['all_outputs'][:1]}") # Uncomment to see sample output

    print("\nEvaluation complete. This simulation demonstrates a basic framework for comparing design patterns.")
    print("For a real scientific paper, you would replace simulations with actual LLM API calls and real tools,")
    print("and use more rigorous statistical analysis and comprehensive benchmarks.")

