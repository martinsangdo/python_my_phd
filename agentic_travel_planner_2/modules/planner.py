class TaskPlanner:
    def decompose(self, query_data: dict) -> list:
        return [
            "search_flights_api",
            "search_hotels_booking",
            "search_restaurants_tripapi",
            "find_attractions",
            "generate_itinerary",
            "estimate_costs"
        ]