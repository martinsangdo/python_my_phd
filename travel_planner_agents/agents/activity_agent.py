from tools.activity_api import search_activities

class ActivityAgent:
    def __init__(self, memory):
        self.memory = memory

    def plan_activities(self, query):
        print("[ActivityAgent] Finding activities...")
        return search_activities(query)