from tools.flight_api import search_flights

class FlightAgent:
    def __init__(self, memory):
        self.memory = memory

    def plan_flight(self, query):
        print("[FlightAgent] Searching for flights...")
        return search_flights(query)