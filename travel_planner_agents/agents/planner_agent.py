from agents.flight_agent import FlightAgent
from agents.hotel_agent import HotelAgent
from agents.activity_agent import ActivityAgent
from agents.budget_agent import BudgetAgent
from memory.memory_store import MemoryStore

class PlannerAgent:
    def __init__(self):
        self.memory = MemoryStore()
        self.flight_agent = FlightAgent(self.memory)
        self.hotel_agent = HotelAgent(self.memory)
        self.activity_agent = ActivityAgent(self.memory)
        self.budget_agent = BudgetAgent(self.memory)

    def run(self, user_request):
        self.memory.store("user_request", user_request)
        #1 Perception & Interpretation:** Parses the query, extracting destination (Thailand), duration (2 weeks), season (winter), traveler count (family implies 4), budget ($5,000), preferences (kid-friendly hotels, cultural activities, elephants).


        #2 Planning: 


        #3: Tool use (execute planned tasks): 

        #5: Perception: summarize data

        #6: If no schedule available, give more suggestions to user to select: increase budget, change dates, etc

        #7: adaptive response: e planing


        flights = self.flight_agent.plan_flight(user_request)
        hotels = self.hotel_agent.plan_hotel(user_request)
        activities = self.activity_agent.plan_activities(user_request)

        total_cost = self.budget_agent.evaluate(flights, hotels, activities)
        plan = {
            "flights": flights,
            "hotels": hotels,
            "activities": activities,
            "estimated_cost": total_cost
        }

        self.memory.store("final_plan", plan)
        return plan