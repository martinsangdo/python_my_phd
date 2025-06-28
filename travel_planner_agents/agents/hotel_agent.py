from tools.hotel_api import search_hotels

class HotelAgent:
    def __init__(self, memory):
        self.memory = memory

    def plan_hotel(self, query):
        print("[HotelAgent] Searching for hotels...")
        return search_hotels(query)