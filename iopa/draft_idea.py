Here is the Python code for the **Intelligent Orchestration & Proactive Adaptation (IOPA)** hybrid agentic AI design pattern.

This code provides a **conceptual framework** demonstrating the interaction between the various modules of the IOPA agent. It uses:

  * `asyncio` for asynchronous operations.
  * **Simulated MCP client interactions** for tool calls and resource subscriptions, as a full MCP SDK and server setup would be extensive.
  * `langchain` for LLM integration (using `ChatOpenAI` as an example). **You will need to replace `"your_openai_api_key"` with a valid OpenAI API key to run this part.**
  * `chromadb` for the vector database (memory module).
  * A simple `asyncio.Queue` for the internal event bus.

**To run this code:**

1.  **Install necessary libraries:**
    ```bash
    pip install langchain-openai chromadb sentence-transformers
    ```
2.  **Replace `"your_openai_api_key"`** in the `LLM_API_KEY` variable with your actual OpenAI API key.
3.  Execute the Python script.

This implementation focuses on the architectural flow and the roles of each module within the IOPA pattern. For a production system, each module would be significantly more complex, involving robust error handling, detailed data models, and actual integrations with external services via real MCP servers.

```python
import asyncio
from collections import deque
from datetime import datetime, timedelta
import json
import logging
from typing import Dict, Any, List, Optional, Callable, Awaitable

# --- External Libraries/Frameworks (Conceptual Imports) ---
# Assuming 'fastmcp' is an MCP Python SDK for client-side interactions
# You'd need to install it: pip install fastmcp (if available)
# For this conceptual code, we'll mock the MCPClient interactions.
# In a real scenario, you'd import from a library like 'fastmcp.client'
# from fastmcp.client import MCPClient, ToolCall, ResourceSubscription, Event

# Mocking MCPClient and its types for conceptual demonstration
class MockToolCall:
    def __init__(self, tool_name: str, parameters: Dict[str, Any]):
        self.tool_name = tool_name
        self.parameters = parameters

class MockResourceSubscription:
    def __init__(self, name: str, uri: str, callback: Callable[[Any], Awaitable[None]]):
        self.name = name
        self.uri = uri
        self.callback = callback

class MockEvent:
    def __init__(self, data: Any):
        self.data = data

class MockMCPClient:
    """A mock MCP Client to simulate interactions without a real MCP server setup."""
    def __init__(self):
        self._subscribed_resources: Dict[str, MockResourceSubscription] = {}
        self._tool_handlers: Dict[str, Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]] = {}
        self.event_queue = asyncio.Queue() # For simulating incoming MCP events

    async def invoke_tool(self, tool_call: MockToolCall) -> Dict[str, Any]:
        """Simulates invoking a tool on an MCP server."""
        logger.info(f"MockMCPClient: Invoking tool '{tool_call.tool_name}' with params: {tool_call.parameters}")
        handler = self._tool_handlers.get(tool_call.tool_name)
        if handler:
            return await handler(tool_call.parameters)
        return {"status": "error", "message": f"Mock tool '{tool_call.tool_name}' not found."}

    def subscribe_resource(self, name: str, uri: str, callback: Callable[[MockEvent], Awaitable[None]]):
        """Simulates subscribing to an MCP resource."""
        self._subscribed_resources[name] = MockResourceSubscription(name, uri, callback)
        logger.info(f"MockMCPClient: Subscribed to resource '{name}' at '{uri}'")

    # Internal method to simulate resource updates being pushed by a mock server
    async def _simulate_resource_push(self, resource_name: str, data: Any):
        if resource_name in self._subscribed_resources:
            event = MockEvent(data)
            await self._subscribed_resources[resource_name].callback(event)
            logger.info(f"MockMCPClient: Simulating resource push for '{resource_name}' with data: {data}")
        else:
            logger.warning(f"MockMCPClient: No subscription found for resource '{resource_name}' to push data.")

    # Method to register mock tool handlers for the ActionExecutionModule to call
    def register_mock_tool_handler(self, tool_name: str, handler: Callable[[Dict[str, Any]], Awaitable[Dict[str, Any]]]):
        self._tool_handlers[tool_name] = handler
        logger.debug(f"MockMCPClient: Registered mock handler for tool '{tool_name}'")

# Use our mock client
MCPClient = MockMCPClient
ToolCall = MockToolCall
ResourceSubscription = MockResourceSubscription
Event = MockEvent


# LLM integration (e.g., from LangChain or direct API client)
from langchain_core.messages import HumanMessage, SystemMessage, AIMessage
from langchain_core.prompts import ChatPromptTemplate
from langchain_openai import ChatOpenAI # Example LLM

# Vector Database (e.g., ChromaDB)
import chromadb
from chromadb.utils import embedding_functions

# --- Configuration ---
# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger('IOPA_Agent')

# LLM Configuration
LLM_MODEL = "gpt-4o" # Example model
LLM_API_KEY = "your_openai_api_key" # <--- IMPORTANT: REPLACE WITH YOUR ACTUAL OPENAI API KEY

# Vector DB Configuration
VECTOR_DB_PATH = "./chroma_db_iopa" # A unique path for this example
EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2" # Example Sentence-BERT model for embeddings
EMBEDDING_FUNCTION = embedding_functions.SentenceTransformerEmbeddingFunction(
    model_name=EMBEDDING_MODEL_NAME
)

# MCP Server Endpoints (These would be the URLs/addresses of your real MCP Servers)
# For this mock, these are just identifiers.
MCP_SERVER_ENDPOINTS = {
    "hotel_booking": "mcp://hotel_service/api",
    "flight_booking": "mcp://flight_service/api",
    "attractions": "mcp://attractions_service/api",
    "weather": "mcp://weather_service/api",
}

# --- Shared Utilities / Types ---
class AgentState:
    """Represents the current internal state of the agent."""
    def __init__(self, user_id: str):
        self.user_id = user_id
        self.conversation_history: List[Dict[str, Any]] = [] # For short-term memory
        self.current_goal: Optional[Dict[str, Any]] = None
        self.active_plan: List[Dict[str, Any]] = [] # Sequence of steps/sub-goals
        self.constraints: Dict[str, Any] = {} # e.g., budget, dates, preferences
        self.realtime_alerts: List[Dict[str, Any]] = [] # From proactive monitoring
        self.user_preferences: Dict[str, Any] = {} # Learned preferences
        self.internal_thoughts: List[str] = [] # For reflection
        self.pending_actions: List[Dict[str, Any]] = deque() # Actions to be executed by Action/Execution module

class TravelPlan:
    """A structured representation of a travel itinerary."""
    def __init__(self, destination: str, start_date: datetime, end_date: datetime):
        self.destination = destination
        self.start_date = start_date
        self.end_date = end_date
        self.flights: List[Dict[str, Any]] = []
        self.hotels: List[Dict[str, Any]] = []
        self.activities: List[Dict[str, Any]] = []
        self.status: str = "draft" # draft, pending_booking, booked, cancelled

class EventBus:
    """Simple in-memory event bus using asyncio.Queue."""
    def __init__(self):
        self._subscribers: Dict[str, List[Queue]] = {}

    async def publish(self, event_type: str, data: Any):
        """Publishes an event to all subscribed queues."""
        if event_type in self._subscribers:
            for queue in self._subscribers[event_type]:
                await queue.put(data)
            logger.info(f"Event published: {event_type} - {data}")
        else:
            logger.debug(f"No subscribers for event type: {event_type}")

    async def subscribe(self, event_type: str, handler: Callable[[Any], Awaitable[None]]):
        """Subscribes a handler to an event type."""
        queue = Queue()
        if event_type not in self._subscribers:
            self._subscribers[event_type] = []
        self._subscribers[event_type].append(queue)
        logger.debug(f"Subscribed handler to event type: {event_type}")

        while True:
            data = await queue.get()
            try:
                await handler(data)
            except Exception as e:
                logger.error(f"Error in event handler for {event_type}: {e}")
            finally:
                queue.task_done()

event_bus = EventBus()

# --- 1. Perception & Interpretation Module ---
class PerceptionModule:
    def __init__(self, agent_state: AgentState, llm_client):
        self.agent_state = agent_state
        self.llm = llm_client
        self.intent_extraction_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are an expert travel assistant. Your task is to extract the user's "
                "travel intent, destination, dates, budget (numeric, e.g., 3000), number of travelers (numeric), and any "
                "specific preferences from their query. Respond with a JSON object."
                "Example: {'intent': 'plan_trip', 'destination': 'London', 'start_date': '2025-08-01', 'end_date': '2025-08-07', 'budget': 2000, 'travelers': 2, 'preferences': ['sightseeing', 'good food']}"
                "If information is missing, infer or leave as null if truly unknown. Dates should be in YYYY-MM-DD format. Prioritize key entities."
                "Current Date: " + datetime.now().strftime("%Y-%m-%d") # Provide current date for relative date parsing
            ),
            HumanMessage(content="{query}")
        ])

    async def process_user_query(self, user_query: str) -> Dict[str, Any]:
        logger.info(f"Perception: Processing user query: {user_query}")
        self.agent_state.conversation_history.append({"role": "user", "content": user_query})

        # Use LLM for intent & entity extraction
        try:
            llm_response = await self.llm.invoke(self.intent_extraction_prompt.format(query=user_query))
            extracted_info = json.loads(llm_response.content)
            logger.info(f"Perception: Extracted info: {extracted_info}")
            # Update agent state
            self.agent_state.current_goal = extracted_info
            # Merge constraints
            self.agent_state.constraints.update({k: v for k, v in extracted_info.items() if k not in ['intent', 'preferences']})
            # Update user preferences in agent state
            if 'preferences' in extracted_info:
                for pref in extracted_info['preferences']:
                    self.agent_state.user_preferences[pref] = True # Simple boolean preference

            return extracted_info
        except json.JSONDecodeError as e:
            logger.error(f"Perception: LLM response not valid JSON: {llm_response.content} - {e}")
            return {"intent": "fallback", "error": "Invalid LLM response format."}
        except Exception as e:
            logger.error(f"Perception: Error extracting intent: {e}")
            return {"intent": "fallback", "error": str(e)}

    async def process_realtime_event(self, event_data: Dict[str, Any]):
        """Processes real-time updates from MCP Resources."""
        logger.info(f"Perception: Processing real-time event: {event_data}")
        self.agent_state.realtime_alerts.append(event_data)
        await event_bus.publish("realtime_alert_received", event_data) # Notify orchestrator


# --- 2. Planning & Decomposition Module ---
class PlanningModule:
    def __init__(self, agent_state: AgentState, llm_client):
        self.agent_state = agent_state
        self.llm = llm_client
        self.planning_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a sophisticated travel planning AI. Given the user's goal, "
                "current state, and available tools, generate a step-by-step plan "
                "to achieve the goal. Each step should be an actionable sub-goal or a tool call. "
                "Consider efficiency, constraints, and current alerts. Output as a JSON list of steps."
                "Example: [\n"
                "  {'task': 'search_flights', 'tool_name': 'find_flights', 'params': {'destination': 'London', 'departure_date': '...', 'return_date': '...', 'num_passengers': 2}},\n"
                "  {'task': 'search_hotels', 'tool_name': 'search_hotels', 'params': {'destination': 'London', 'check_in_date': '...', 'check_out_date': '...', 'min_stars': 4}},\n"
                "  {'task': 'evaluate_options', 'category': 'flights_hotels', 'description': 'Synthesize flight and hotel options'},\n"
                "  {'task': 'propose_itinerary', 'description': 'Propose the plan to the user'}\n"
                "]"
                "\nAvailable tools (JSON schema, use 'tool_name' and 'parameters'): {tool_schemas_json}\n"
                "Current Agent State summary: {agent_state_summary}\n"
                "Real-time Alerts: {realtime_alerts_summary}\n"
                "User Preferences: {user_preferences_summary}\n"
                "Strictly output only the JSON list."
            ),
            HumanMessage(content="User Goal: {goal}")
        ])

    async def generate_plan(self, tool_schemas: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        logger.info(f"Planning: Generating plan for goal: {self.agent_state.current_goal}")
        tool_schemas_json = json.dumps(tool_schemas)
        agent_state_summary = json.dumps({
            "current_goal": self.agent_state.current_goal,
            "constraints": self.agent_state.constraints,
            "internal_thoughts": self.agent_state.internal_thoughts[-2:] # Last 2 thoughts
        })
        realtime_alerts_summary = json.dumps(self.agent_state.realtime_alerts)
        user_preferences_summary = json.dumps(self.agent_state.user_preferences)

        try:
            llm_response = await self.llm.invoke(self.planning_prompt.format(
                tool_schemas_json=tool_schemas_json,
                agent_state_summary=agent_state_summary,
                realtime_alerts_summary=realtime_alerts_summary,
                user_preferences_summary=user_preferences_summary,
                goal=self.agent_state.current_goal
            ))
            plan_steps = json.loads(llm_response.content)
            self.agent_state.active_plan = plan_steps
            logger.info(f"Planning: Generated plan: {plan_steps}")
            return plan_steps
        except json.JSONDecodeError as e:
            logger.error(f"Planning: LLM response not valid JSON: {llm_response.content} - {e}")
            return [{"task": "error", "message": f"Failed to parse plan from LLM: {e}"}]
        except Exception as e:
            logger.error(f"Planning: Error generating plan: {e}")
            return [{"task": "error", "message": f"Failed to generate plan: {e}"}]

    async def re_plan(self, feedback: Dict[str, Any]):
        logger.warning(f"Planning: Re-planning triggered by feedback: {feedback}")
        # Incorporate feedback into agent state for next planning iteration
        self.agent_state.internal_thoughts.append(f"Re-planning due to feedback: {json.dumps(feedback)}")
        # Clear old alerts after processing them if they were the cause of re-plan
        # self.agent_state.realtime_alerts = [] # Keep alerts for next plan iteration if still relevant
        # Re-trigger plan generation
        await event_bus.publish("plan_needed", self.agent_state.current_goal)


# --- 3. Memory & Knowledge Base Module ---
class MemoryModule:
    def __init__(self):
        self.chroma_client = chromadb.PersistentClient(path=VECTOR_DB_PATH)
        self.collections = {
            "episodic_memory": self.chroma_client.get_or_create_collection(
                name="episodic_memory", embedding_function=EMBEDDING_FUNCTION
            ),
            "semantic_knowledge": self.chroma_client.get_or_create_collection(
                name="semantic_knowledge", embedding_function=EMBEDDING_FUNCTION
            ),
            "user_preferences": self.chroma_client.get_or_create_collection(
                name="user_preferences", embedding_function=EMBEDDING_FUNCTION
            )
        }
        self.feature_store: Dict[str, Any] = {} # In-memory simple feature store for real-time data

    async def add_episodic_memory(self, user_id: str, event_description: str, metadata: Dict[str, Any]):
        """Stores past interactions/events."""
        logger.info(f"Memory: Adding episodic memory for {user_id}: {event_description}")
        await asyncio.to_thread(self.collections["episodic_memory"].add,
                                documents=[event_description],
                                metadatas=[{"user_id": user_id, "timestamp": datetime.now().isoformat(), **metadata}],
                                ids=[f"epi_{user_id}_{datetime.now().timestamp()}"])

    async def retrieve_episodic_memory(self, user_id: str, query: str, n_results: int = 5) -> List[str]:
        """Retrieves relevant past interactions."""
        logger.info(f"Memory: Retrieving episodic memory for {user_id} with query: {query}")
        results = await asyncio.to_thread(self.collections["episodic_memory"].query,
                                           query_texts=[query],
                                           n_results=n_results,
                                           where={"user_id": user_id})
        return [doc for doc in results['documents']] if results and results['documents'] else []

    async def update_user_preference(self, user_id: str, preference_key: str, preference_value: Any):
        """Updates user preferences (could be complex, e.g., learned from behavior)."""
        pref_doc = f"User {user_id} prefers {preference_key}: {preference_value}"
        await asyncio.to_thread(self.collections["user_preferences"].upsert,
                                documents=[pref_doc],
                                metadatas=[{"user_id": user_id, "key": preference_key, "timestamp": datetime.now().isoformat()}],
                                ids=[f"pref_{user_id}_{preference_key}"])
        logger.info(f"Memory: Updated user preference for {user_id}: {preference_key}={preference_value}")

    async def get_user_preferences(self, user_id: str, query: str = "", n_results: int = 3) -> Dict[str, Any]:
        """Retrieves user preferences."""
        results = await asyncio.to_thread(self.collections["user_preferences"].query,
                                           query_texts=[query if query else "user preferences"],
                                           n_results=n_results,
                                           where={"user_id": user_id})
        prefs = {}
        if results and results['metadatas']:
            for meta, doc_list in zip(results['metadatas'], results['documents']):
                if doc_list: # Ensure doc_list is not empty
                    doc = doc_list[0] # Take the first document if it's a list of lists
                    key = meta[0].get('key') # Access first metadata dict
                    if key and ":" in doc:
                        value = doc.split(":", 1)[1].strip()
                        prefs[key] = value # Simple parsing, could be more robust
        logger.info(f"Memory: Retrieved preferences for {user_id}: {prefs}")
        return prefs

    def update_realtime_feature(self, feature_key: str, data: Any):
        """Updates a real-time feature in the in-memory store."""
        self.feature_store[feature_key] = {"data": data, "timestamp": datetime.now()}
        logger.debug(f"Memory: Updated real-time feature: {feature_key}")

    def get_realtime_feature(self, feature_key: str) -> Optional[Any]:
        """Retrieves a real-time feature."""
        feature = self.feature_store.get(feature_key)
        if feature:
            logger.debug(f"Memory: Retrieved real-time feature: {feature_key}")
            return feature['data']
        return None

# --- 4. Tool Orchestrator & Dynamic Selection Module ---
class ToolOrchestrator:
    def __init__(self, mcp_client: MCPClient, agent_state: AgentState, llm_client):
        self.mcp_client = mcp_client
        self.agent_state = agent_state
        self.llm = llm_client
        self.available_tools: Dict[str, Any] = {} # Populated from MCP discovery
        self.tool_selection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are an expert in selecting the best tool for a given task. "
                "Given the current task and available MCP tools, generate a JSON object "
                "representing the tool call. Include the 'tool_name' and 'parameters'. "
                "Parameters should be extracted precisely from the task and current agent state. "
                "Ensure all required parameters for the selected tool are present. "
                "If a parameter value is missing, try to infer it from the agent state or leave it as null if truly unknown. "
                "If no suitable tool is found, return {'tool_name': 'no_tool_found', 'parameters': {}}."
                "MCP Tools available (JSON schema): {tool_schemas_json}\n"
                "Current Agent State summary: {agent_state_summary}\n"
                "User Preferences (for context): {user_preferences_summary}\n"
                "Real-time Alerts (for context): {realtime_alerts_summary}\n"
                "Strictly output only the JSON object."
            ),
            HumanMessage(content="Current Task: {task_description}")
        ])

    async def discover_tools(self):
        """Discovers tools from connected MCP servers."""
        for name, endpoint in MCP_SERVER_ENDPOINTS.items():
            try:
                logger.info(f"ToolOrchestrator: Simulating discovery for {name} at {endpoint}")
                # Mock tool schemas for demonstration
                if name == "hotel_booking":
                    self.available_tools["search_hotels"] = {
                        "name": "search_hotels",
                        "description": "Searches for hotels based on criteria.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "destination": {"type": "string"},
                                "check_in_date": {"type": "string", "format": "date"},
                                "check_out_date": {"type": "string", "format": "date"},
                                "min_stars": {"type": "integer", "default": 3},
                                "max_price_per_night": {"type": "number"},
                                "kid_friendly": {"type": "boolean", "default": False},
                            },
                            "required": ["destination", "check_in_date", "check_out_date"]
                        },
                        "semantic_tags": ["accommodation", "lodging", "family_travel"]
                    }
                elif name == "flight_booking":
                    self.available_tools["find_flights"] = {
                        "name": "find_flights",
                        "description": "Finds flights between two locations.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "origin": {"type": "string"},
                                "destination": {"type": "string"},
                                "departure_date": {"type": "string", "format": "date"},
                                "return_date": {"type": "string", "format": "date"},
                                "num_passengers": {"type": "integer", "default": 1},
                                "max_price": {"type": "number"},
                            },
                            "required": ["origin", "destination", "departure_date"]
                        },
                        "semantic_tags": ["transportation", "air_travel"]
                    }
                    self.available_tools["get_flight_status"] = {
                        "name": "get_flight_status",
                        "description": "Gets real-time status of a flight.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "flight_number": {"type": "string"},
                                "flight_date": {"type": "string", "format": "date"},
                            },
                            "required": ["flight_number", "flight_date"]
                        },
                        "semantic_tags": ["realtime", "status"]
                    }
                elif name == "weather":
                    self.available_tools["get_weather_forecast"] = {
                        "name": "get_weather_forecast",
                        "description": "Gets weather forecast for a location and date.",
                        "parameters": {
                            "type": "object",
                            "properties": {
                                "location": {"type": "string"},
                                "date": {"type": "string", "format": "date"},
                            },
                            "required": ["location", "date"]
                        },
                        "semantic_tags": ["environmental", "realtime"]
                    }
                    # Also register a resource for proactive monitoring
                    # In a real MCP setup, the MCPClient would handle the subscription to the server
                    # For mock, we just tell the mock client to simulate the callback.
                    self.mcp_client.subscribe_resource(
                        f"weather_alerts_{name}",
                        f"mcp://{name}/weather_alerts",
                        self._handle_weather_alert_resource_update
                    )
                logger.info(f"ToolOrchestrator: Discovered tools from {name}: {list(self.available_tools.keys())}")
            except Exception as e:
                logger.error(f"ToolOrchestrator: Failed to discover tools from {name}: {e}")

    async def _handle_weather_alert_resource_update(self, event: Event):
        """Callback for real-time weather alerts from an MCP Resource."""
        logger.info(f"ToolOrchestrator: Real-time weather alert received from MCP Resource: {event.data}")
        await event_bus.publish("realtime_event", {"type": "weather_alert", "data": event.data})

    async def select_and_prepare_tool_call(self, task: Dict[str, Any]) -> Optional[ToolCall]:
        """Uses LLM to select tool and prepare parameters."""
        logger.info(f"ToolOrchestrator: Selecting tool for task: {task}")
        tool_schemas = list(self.available_tools.values())
        tool_schemas_json = json.dumps(tool_schemas)
        agent_state_summary = json.dumps({
            "current_goal": self.agent_state.current_goal,
            "constraints": self.agent_state.constraints,
            "internal_thoughts": self.agent_state.internal_thoughts[-2:] if self.agent_state.internal_thoughts else []
        })
        user_preferences_summary = json.dumps(self.agent_state.user_preferences)
        realtime_alerts_summary = json.dumps(self.agent_state.realtime_alerts)

        task_description = json.dumps(task) # Pass the full task dictionary

        try:
            llm_response = await self.llm.invoke(self.tool_selection_prompt.format(
                tool_schemas_json=tool_schemas_json,
                agent_state_summary=agent_state_summary,
                user_preferences_summary=user_preferences_summary,
                realtime_alerts_summary=realtime_alerts_summary,
                task_description=task_description
            ))
            tool_call_data = json.loads(llm_response.content)
            tool_name = tool_call_data.get("tool_name")
            parameters = tool_call_data.get("parameters", {})

            if tool_name and tool_name in self.available_tools:
                # Basic validation (could be more rigorous with Pydantic)
                required_params = self.available_tools[tool_name]['parameters'].get('required', [])
                for param in required_params:
                    if param not in parameters or parameters[param] is None:
                        logger.error(f"ToolOrchestrator: Missing required parameter '{param}' for {tool_name}. "
                                     f"Required: {required_params}, Given: {parameters}")
                        raise ValueError(f"Missing required parameter '{param}' for tool call.")

                return ToolCall(tool_name=tool_name, parameters=parameters)
            elif tool_name == "no_tool_found":
                logger.info("ToolOrchestrator: LLM indicated no suitable tool found.")
                return None
            else:
                logger.error(f"ToolOrchestrator: LLM proposed unknown tool or invalid format: {tool_call_data}")
                return None
        except json.JSONDecodeError as e:
            logger.error(f"ToolOrchestrator: LLM response not valid JSON for tool selection: {llm_response.content} - {e}")
            return None
        except Exception as e:
            logger.error(f"ToolOrchestrator: Error in tool selection: {e}")
            return None

# --- 5. Action & Execution Module ---
class ActionExecutionModule:
    def __init__(self, mcp_client: MCPClient, agent_state: AgentState):
        self.mcp_client = mcp_client
        self.agent_state = agent_state
        # Register mock handlers for the MCP client to call
        self.mcp_client.register_mock_tool_handler("search_hotels", self._mock_search_hotels)
        self.mcp_client.register_mock_tool_handler("find_flights", self._mock_find_flights)
        self.mcp_client.register_mock_tool_handler("get_flight_status", self._mock_get_flight_status)
        self.mcp_client.register_mock_tool_handler("get_weather_forecast", self._mock_get_weather_forecast)


    async def _mock_search_hotels(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates a hotel search API call."""
        destination = params.get("destination", "unknown")
        min_stars = params.get("min_stars", 3)
        max_price = params.get("max_price_per_night", 300)
        kid_friendly = params.get("kid_friendly", False)

        if "error" in destination.lower(): # Simulate failure
            return {"status": "failed", "error": "Simulated API error: destination not found."}

        hotels = [
            {"name": f"Grand {destination} Hotel", "price": max_price - 50, "stars": 5, "kid_friendly": True},
            {"name": f"Cozy Stay {destination}", "price": max_price - 150, "stars": min_stars, "kid_friendly": False},
        ]
        filtered_hotels = [h for h in hotels if h["stars"] >= min_stars and h["price"] <= max_price and (not kid_friendly or h["kid_friendly"])]
        await asyncio.sleep(0.5) # Simulate network delay
        return {"status": "success", "results": filtered_hotels}

    async def _mock_find_flights(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates a flight search API call."""
        destination = params.get("destination", "unknown")
        origin = params.get("origin", "unknown")
        departure_date = params.get("departure_date")
        num_passengers = params.get("num_passengers", 1)

        if "error" in destination.lower(): # Simulate failure
            return {"status": "failed", "error": "Simulated API error: flight search failed."}

        flights = [
            {"flight_number": "FL123", "price": 450 * num_passengers, "airline": "SkyJet", "departure_time": "08:00"},
            {"flight_number": "FL456", "price": 520 * num_passengers, "airline": "AirWings", "departure_time": "14:00"},
        ]
        await asyncio.sleep(0.7) # Simulate network delay
        return {"status": "success", "results": flights}

    async def _mock_get_flight_status(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates getting flight status."""
        flight_number = params.get("flight_number")
        flight_date = params.get("flight_date")
        await asyncio.sleep(0.2)
        return {"status": "success", "results": {"flight_number": flight_number, "status": "on_time", "gate": "A12"}}

    async def _mock_get_weather_forecast(self, params: Dict[str, Any]) -> Dict[str, Any]:
        """Simulates getting weather forecast."""
        location = params.get("location")
        date = params.get("date")
        await asyncio.sleep(0.3)
        return {"status": "success", "results": {"location": location, "date": date, "forecast": "sunny", "temperature": "25C"}}

    async def execute_tool_call(self, tool_call: ToolCall) -> Dict[str, Any]:
        logger.info(f"ActionExecution: Executing tool call: {tool_call.tool_name} with params {tool_call.parameters}")
        try:
            response = await self.mcp_client.invoke_tool(tool_call)
            logger.info(f"ActionExecution: Tool call result: {response}")
            return response
        except Exception as e:
            logger.error(f"ActionExecution: Error executing tool {tool_call.tool_name}: {e}")
            return {"tool": tool_call.tool_name, "status": "failed", "error": str(e)}

# --- 6. Reflection & Learning Module ---
class ReflectionModule:
    def __init__(self, agent_state: AgentState, llm_client, memory_module: MemoryModule):
        self.agent_state = agent_state
        self.llm = llm_client
        self.memory = memory_module
        self.reflection_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are the reflection component of an AI agent. Review the executed "
                "action, its result, and the current agent state. Identify if the action "
                "was successful, if the goal is closer, and if any re-planning or user "
                "clarification is needed. Provide insights and suggest next steps. "
                "If the action failed, explain why and suggest a retry strategy or alternative."
                "Output a JSON object with 'status': 'ok'/'replan'/'clarify', 'reason': '...', 'suggested_action': '...'"
                "Current Agent State: {agent_state_summary}\n"
                "Executed Action: {action_summary}\n"
                "Action Result: {result_summary}\n"
            ),
            HumanMessage(content="Reflect on this.")
        ])

    async def reflect_on_action(self, action: Dict[str, Any], result: Dict[str, Any]) -> Dict[str, Any]:
        logger.info(f"Reflection: Reflecting on action: {action.get('task')} with result: {result.get('status')}")
        agent_state_summary = json.dumps({
            "current_goal": self.agent_state.current_goal,
            "constraints": self.agent_state.constraints,
            "active_plan_step": action, # Reflect on the current action
            "realtime_alerts": self.agent_state.realtime_alerts
        })
        action_summary = json.dumps(action)
        result_summary = json.dumps(result)

        try:
            llm_response = await self.llm.invoke(self.reflection_prompt.format(
                agent_state_summary=agent_state_summary,
                action_summary=action_summary,
                result_summary=result_summary
            ))
            reflection_output = json.loads(llm_response.content)
            self.agent_state.internal_thoughts.append(f"Reflection: {reflection_output.get('reason', 'No reason provided')}")

            if reflection_output.get('status') == 'replan':
                await event_bus.publish("replan_needed", {"reason": reflection_output.get('reason'), "action": action, "result": result})
            elif reflection_output.get('status') == 'clarify':
                await event_bus.publish("user_clarification_needed", {"question": reflection_output.get('suggested_action')})
            elif reflection_output.get('status') == 'ok':
                pass # Continue with next plan step

            # Update memory based on successful/failed action for learning
            await self.memory.add_episodic_memory(
                user_id=self.agent_state.user_id,
                event_description=f"Action '{action.get('task')}' resulted in '{result.get('status')}'",
                metadata={"action": action, "result": result}
            )

            return reflection_output
        except json.JSONDecodeError as e:
            logger.error(f"Reflection: LLM response not valid JSON for reflection: {llm_response.content} - {e}")
            return {"status": "replan", "reason": f"Reflection failed to parse LLM response: {e}", "suggested_action": "Retry or re-evaluate goal."}
        except Exception as e:
            logger.error(f"Reflection: Error during reflection: {e}")
            return {"status": "replan", "reason": f"Reflection failed: {e}", "suggested_action": "Retry or re-evaluate goal."}

# --- 7. Guardrails & Safety Module ---
class GuardrailsModule:
    def __init__(self, agent_state: AgentState):
        self.agent_state = agent_state
        # Define simple rules. In production, use a dedicated rule engine like Durable Rules.
        self.policy_rules = {
            "budget_check": lambda params, state: params.get("max_price") is None or state.constraints.get("budget", float('inf')) is None or params.get("max_price") <= state.constraints.get("budget", float('inf')),
            "sensitive_info_redaction": lambda text: text.replace("credit card", "[REDACTED_CREDIT_CARD_INFO]").replace("password", "[REDACTED_PASSWORD]"),
            "auto_booking_permission": lambda state: state.user_preferences.get("auto_book", False) is True
        }

    async def pre_action_check(self, tool_call: ToolCall) -> bool:
        """Checks policies before executing a tool call."""
        logger.info(f"Guardrails: Pre-action check for {tool_call.tool_name}")

        # Example: Budget check for booking tools
        if tool_call.tool_name in ["find_flights", "search_hotels"]:
            if not self.policy_rules["budget_check"](tool_call.parameters, self.agent_state):
                logger.warning("Guardrails: Budget check failed for tool call.")
                await event_bus.publish("user_clarification_needed", {
                    "question": f"The proposed {tool_call.tool_name.replace('_', ' ')} might exceed your budget of ${self.agent_state.constraints.get('budget', 'N/A')}. Would you like to increase your budget or explore cheaper options?"
                })
                return False

        # Example: Auto-booking confirmation for final booking tools
        if tool_call.tool_name in ["book_flight", "book_hotel"]: # Hypothetical booking tools
            if not self.policy_rules["auto_booking_permission"](self.agent_state):
                logger.warning("Guardrails: Auto-booking permission not granted. Human-in-the-loop required.")
                await event_bus.publish("user_clarification_needed", {
                    "question": f"I'm ready to book your {tool_call.tool_name.replace('_', ' ')}. Please confirm if I should proceed with the booking, as auto-booking is not enabled."
                })
                return False
        
        logger.info(f"Guardrails: Pre-action check passed for {tool_call.tool_name}")
        return True

    async def post_action_filter(self, result: Dict[str, Any]) -> Dict[str, Any]:
        """Filters/sanitizes tool results before passing them back."""
        logger.info(f"Guardrails: Post-action filter for result of {result.get('tool')}")
        if "results" in result and isinstance(result["results"], list):
            for item in result["results"]:
                for key, value in item.items():
                    if isinstance(value, str):
                        item[key] = self.policy_rules["sensitive_info_redaction"](value)
        logger.info(f"Guardrails: Post-action filter complete.")
        return result

# --- 8. Proactive Monitoring & Alerting Subsystem ---
class ProactiveMonitoring:
    def __init__(self, memory_module: MemoryModule, mcp_client: MCPClient):
        self.memory = memory_module
        self.mcp_client = mcp_client
        self.monitored_items: Dict[str, Any] = {} # e.g., {"flight_FL123": {"last_price": 500, "threshold": 0.1}}

    async def start_monitoring(self):
        """Starts listening for real-time events from MCP resources."""
        # This would typically be done by subscribing to MCP resources that stream data.
        # For this mock, we'll simulate events being pushed to the MCPClient,
        # which then triggers the registered callbacks in ToolOrchestrator.
        logger.info("Proactive Monitoring: Initializing monitoring of MCP resources.")
        # Example: Simulating a weather alert push after a delay
        asyncio.create_task(self._simulate_external_resource_pushes())

    async def _simulate_external_resource_pushes(self):
        """Simulates external MCP servers pushing real-time updates."""
        await asyncio.sleep(10) # Wait a bit after startup
        logger.info("Proactive Monitoring: SIMULATING external weather alert push.")
        await self.mcp_client._simulate_resource_push(
            "weather_alerts_weather",
            {"location": "Paris", "alert_info": "Heavy rain expected next week. Consider indoor activities."}
        )
        await asyncio.sleep(15)
        logger.info("Proactive Monitoring: SIMULATING external flight price change push.")
        await self.mcp_client._simulate_resource_push(
            "flight_price_alerts_flight_booking", # Hypothetical resource name
            {"item_id": "flight_to_paris", "new_price": 280, "last_price": 350, "threshold": 0.1}
        )


# --- Main Orchestrator (The IOPA Agent) ---
class IOPAAgent:
    def __init__(self, user_id: str):
        self.agent_state = AgentState(user_id=user_id)
        self.llm_client = ChatOpenAI(model=LLM_MODEL, openai_api_key=LLM_API_KEY)
        self.mcp_client = MCPClient() # Our mock MCP client

        self.memory_module = MemoryModule()
        self.perception_module = PerceptionModule(self.agent_state, self.llm_client)
        self.planning_module = PlanningModule(self.agent_state, self.llm_client)
        self.tool_orchestrator = ToolOrchestrator(self.mcp_client, self.agent_state, self.llm_client)
        self.action_execution_module = ActionExecutionModule(self.mcp_client, self.agent_state)
        self.reflection_module = ReflectionModule(self.agent_state, self.llm_client, self.memory_module)
        self.guardrails_module = GuardrailsModule(self.agent_state)
        self.proactive_monitoring = ProactiveMonitoring(self.memory_module, self.mcp_client)

        # Initialize event bus consumers
        asyncio.create_task(event_bus.subscribe("plan_needed", self._handle_plan_needed))
        asyncio.create_task(event_bus.subscribe("replan_needed", self._handle_replan_needed))
        asyncio.create_task(event_bus.subscribe("user_clarification_needed", self._handle_user_clarification))
        asyncio.create_task(event_bus.subscribe("proactive_suggestion", self._handle_proactive_suggestion))
        # Perception module processes raw real-time events from MCP (via ToolOrchestrator callback)
        # and then publishes to 'realtime_alert_received' for internal handling.
        # The ProactiveMonitoring module also starts listening for simulated pushes.
        asyncio.create_task(self.proactive_monitoring.start_monitoring())


    async def initialize(self):
        """Initializes the agent, discovering MCP tools and loading user preferences."""
        logger.info("Agent: Initializing...")
        await self.tool_orchestrator.discover_tools()
        # Load user preferences from long-term memory at start
        self.agent_state.user_preferences = await self.memory_module.get_user_preferences(self.agent_state.user_id, query="all preferences")
        logger.info(f"Agent: Initialized with preferences: {self.agent_state.user_preferences}")
        # Add a dummy preference for testing auto-booking guardrail
        # await self.memory_module.update_user_preference(self.agent_state.user_id, "auto_book", True)


    async def _handle_plan_needed(self, goal: Dict[str, Any]):
        """Handler for 'plan_needed' event."""
        logger.info("Agent: Handling 'plan_needed' event.")
        self.agent_state.current_goal = goal # Ensure current_goal is set for planning module
        plan = await self.planning_module.generate_plan(list(self.tool_orchestrator.available_tools.values()))
        self.agent_state.pending_actions.extend(plan)
        asyncio.create_task(self._execute_next_plan_step()) # Start execution as a separate task

    async def _handle_replan_needed(self, feedback: Dict[str, Any]):
        """Handler for 'replan_needed' event."""
        logger.warning(f"Agent: Handling 'replan_needed' event due to: {feedback.get('reason')}")
        self.agent_state.internal_thoughts.append(f"Re-planning triggered by failure/feedback: {feedback.get('reason')}")
        # Clear current pending actions as plan needs to be regenerated
        self.agent_state.pending_actions.clear()
        plan = await self.planning_module.generate_plan(list(self.tool_orchestrator.available_tools.values()))
        self.agent_state.pending_actions.extend(plan)
        asyncio.create_task(self._execute_next_plan_step()) # Resume execution

    async def _handle_user_clarification(self, query_data: Dict[str, Any]):
        """Handler for 'user_clarification_needed' event."""
        question = query_data.get('question', 'I need more information to proceed. Can you elaborate?')
        logger.info(f"Agent: Clarification needed: {question}")
        self.agent_state.conversation_history.append({"role": "assistant", "content": f"Assistant (clarification): {question}"})
        print(f"\nAGENT TO USER: {question}\n")
        # In a real system, this would pause the agent's internal loop until user provides input.
        # For this demo, it just prints and continues, but a UI would handle the pause/resume.

    async def _handle_proactive_suggestion(self, suggestion_data: Dict[str, Any]):
        """Handler for 'proactive_suggestion' event."""
        message = suggestion_data.get('message')
        impact = suggestion_data.get('impact')
        logger.info(f"Agent: Proactive suggestion: {message}")
        self.agent_state.conversation_history.append({"role": "assistant", "content": f"Assistant (proactive): {message}"})
        print(f"\nAGENT TO USER (PROACTIVE): {message}\n")
        self.agent_state.internal_thoughts.append(f"Proactive suggestion made: {message}, Impact: {impact}")
        # This would typically prompt the user for a response, which would then be handled by handle_user_input

    async def _execute_next_plan_step(self):
        """Executes the next step in the active plan."""
        while self.agent_state.pending_actions:
            current_task = self.agent_state.pending_actions.popleft() # Use popleft for deque
            logger.info(f"Agent: Executing plan step: {current_task['task']}")

            if current_task['task'] == "evaluate_options":
                await self._evaluate_current_plan_state(current_task)
                continue # Continue to next task in deque

            if current_task['task'] == "propose_itinerary":
                await self._generate_final_response()
                continue # Continue to next task in deque

            # Assume other tasks are tool calls
            tool_call = await self.tool_orchestrator.select_and_prepare_tool_call(current_task)
            if not tool_call:
                logger.error(f"Agent: Could not prepare tool call for task: {current_task}. Re-planning.")
                await event_bus.publish("replan_needed", {"reason": "Failed to select or prepare tool call.", "task": current_task})
                return # Stop current execution, wait for re-plan

            # Guardrails pre-check
            if not await self.guardrails_module.pre_action_check(tool_call):
                logger.warning("Agent: Pre-action guardrail failed. Aborting current tool execution.")
                # user_clarification_needed might have been triggered, waiting for user input
                return # Stop current execution, wait for user input/re-plan

            result = await self.action_execution_module.execute_tool_call(tool_call)

            # Guardrails post-check
            result = await self.guardrails_module.post_action_filter(result)

            # Reflection
            reflection_outcome = await self.reflection_module.reflect_on_action(current_task, result)

            if reflection_outcome.get('status') != 'ok':
                logger.warning(f"Agent: Reflection indicated non-OK status: {reflection_outcome.get('status')}. Stopping plan execution for re-evaluation.")
                # Replan or clarification events would have been published by reflection_module
                return # Stop current execution, wait for event bus handler

            # If all good, proceed to next step in loop
            await asyncio.sleep(0.1) # Yield control

        logger.info("Agent: All plan steps executed or no more pending actions.")
        if not self.agent_state.active_plan and not self.agent_state.pending_actions:
            # If plan completed successfully and no more actions, generate final response if not already done
            if not any(msg.get("content", "").startswith("AGENT:") for msg in self.agent_state.conversation_history[-3:]): # Simple check to avoid double response
                await self._generate_final_response()


    async def _evaluate_current_plan_state(self, task_data: Dict[str, Any]):
        """Internal step to evaluate results of previous tool calls."""
        logger.info(f"Agent: Evaluating current plan state for {task_data.get('category')}")
        
        # Retrieve relevant episodic memories (tool results)
        recent_tool_results = await self.memory_module.retrieve_episodic_memory(
            self.agent_state.user_id,
            query="recent tool call results",
            n_results=5
        )
        
        eval_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "Based on the following information, summarize the travel options found so far "
                "for the user's goal. Indicate if options meet criteria or if further action is needed. "
                "Current Goal: {goal}\n"
                "Constraints: {constraints}\n"
                "Real-time Alerts: {alerts}\n"
                "Recent Tool Results: {recent_results}\n"
                "User Preferences: {preferences}\n"
                "Strictly output a clear summary for the user and indicate next steps (e.g., 'Ready to propose', 'Need more info')."
            ),
            HumanMessage(content="Evaluate the current travel plan progress.")
        ])
        
        response = await self.llm_client.invoke(eval_prompt.format(
            goal=self.agent_state.current_goal,
            constraints=self.agent_state.constraints,
            alerts=self.agent_state.realtime_alerts,
            recent_results=json.dumps(recent_tool_results),
            preferences=self.agent_state.user_preferences
        ))
        self.agent_state.internal_thoughts.append(f"Evaluation of {task_data.get('category')}: {response.content}")
        logger.info(f"Agent: Evaluation complete: {response.content[:100]}...")
        
        # Based on evaluation, you might decide to either add more tasks to pending_actions
        # or signal completion to generate final response.
        # For demo, we'll assume it leads to 'propose_itinerary' if successful.
        if "Ready to propose" in response.content: # Simple keyword check
            self.agent_state.pending_actions.append({"task": "propose_itinerary", "description": "Final proposal"})


    async def _generate_final_response(self):
        """Generates the final response to the user after plan completion."""
        logger.info("Agent: Generating final response.")
        
        # Retrieve relevant information from memory for the final summary
        episodic_summary = await self.memory_module.retrieve_episodic_memory(
            self.agent_state.user_id,
            query="summary of the trip planning process and results",
            n_results=5
        )

        final_response_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(
                "You are a helpful travel planning assistant. Based on the "
                "conversation history, the plan executed, and the results obtained, "
                "provide a comprehensive and friendly summary of the travel plan. "
                "Include key details (destination, dates, budget, travelers, main findings for flights/hotels/activities). "
                "Also, mention any real-time alerts that were handled. "
                "Conclude with any remaining questions or clear next steps for the user (e.g., 'Ready to book?')."
                "Conversation History: {history}\n"
                "Final Plan State Summary (from internal thoughts and memory): {final_state_summary}\n"
                "User Preferences: {preferences}\n"
                "Real-time Alerts Processed: {alerts_processed}\n"
            ),
            HumanMessage(content="Summarize the travel plan.")
        ])
        
        final_state_summary_text = "\n".join(self.agent_state.internal_thoughts) + "\n" + "\n".join(episodic_summary)
        
        llm_response = await self.llm_client.invoke(final_response_prompt.format(
            history=self.agent_state.conversation_history,
            final_state_summary=final_state_summary_text,
            preferences=self.agent_state.user_preferences,
            alerts_processed=json.dumps(self.agent_state.realtime_alerts)
        ))
        final_message = llm_response.content
        self.agent_state.conversation_history.append({"role": "assistant", "content": final_message})
        print(f"\nAGENT: {final_message}\n")

    async def handle_user_input(self, user_input: str):
        """Main entry point for user interaction."""
        logger.info(f"Agent: Received user input: {user_input}")
        extracted_info = await self.perception_module.process_user_query(user_input)

        if extracted_info.get("intent") == "plan_trip":
            await event_bus.publish("plan_needed", extracted_info)
        elif extracted_info.get("intent") == "modify_plan":
            # This would trigger re-planning with new constraints
            self.agent_state.constraints.update(extracted_info)
            await event_bus.publish("replan_needed", {"reason": "User modified plan", "new_constraints": extracted_info})
        elif extracted_info.get("intent") == "confirm_booking":
            print("AGENT: Thank you for confirming! Initiating booking process...")
            # This would trigger a specific booking tool call, potentially with more guardrails
            # For demo, we'll just acknowledge.
        else:
            print("AGENT: I'm not sure how to help with that. Please ask about planning a trip or modifying an existing plan.")


# --- Main Execution ---
async def main():
    user_id = "demo_user_001"
    agent = IOPAAgent(user_id)
    await agent.initialize()

    print("--- Starting IOPA Agent Demo ---")
    print("Type your travel planning requests. Type 'exit' to quit.")

    # Start the main agent loop in the background
    # This loop will process events and execute plan steps
    agent_loop_task = asyncio.create_task(agent._execute_next_plan_step())

    # Simulate user interactions
    user_inputs = [
        "Plan a 7-day trip to Paris next month for 2 people, budget around $3000, looking for a 4-star hotel and some cultural activities.",
        "How about a hotel near the Louvre and a flight from New York?",
        "Actually, let's make the budget $3500 and ensure it's kid-friendly."
    ]

    for user_input in user_inputs:
        print(f"\nUSER: {user_input}")
        await agent.handle_user_input(user_input)
        await asyncio.sleep(15) # Give the agent time to process and respond

    print("\n--- Demo complete. Agent will continue running for proactive alerts. ---")
    # Keep the main loop running to allow proactive alerts to come through
    await asyncio.sleep(30) # Keep running for a bit to observe proactive alerts

if __name__ == "__main__":
    try:
        asyncio.run(main())
    except KeyboardInterrupt:
        logger.info("Agent stopped by user (KeyboardInterrupt).")
    except Exception as e:
        logger.exception("An error occurred during agent execution:")

```