from llm_client import GeminiLLM
from redis_memory import MemoryModule

class ToolOrchestrator:
    def __init__(self):
        self.llm = GeminiLLM()
        self.memory = MemoryModule()

    def execute_plan(self, task_plan: list, query_data: dict, user_id="default_user") -> dict:
        self.memory.save_preferences(user_id, query_data)

        results = {}

        user_context = f"""
        Plan a trip to {query_data['destination']} for {query_data['num_people']} people,
        from {query_data['start_date']} to {query_data['end_date']}.
        Budget: {query_data.get('budget', 'unspecified')} USD.
        Preferences: {query_data['preferences']}
        Accommodation: {query_data['accommodation_style']}, Pace: {query_data['pace']}
        """

        for task in task_plan:
            if task == "search_flights_api":
                results["flights"] = {
                    "airline": "SampleAir",
                    "price_per_person": 480,
                    "total": 480 * query_data["num_people"],
                    "alternative": {
                        "airline": "BudgetFly",
                        "price_per_person": 410,
                        "total": 410 * query_data["num_people"]
                    }
                }

            elif task == "search_hotels_booking":
                nights = query_data['duration_days'] - 1
                style = query_data.get("accommodation_style", "boutique")

                if style == "budget":
                    price_range = (50, 100)
                elif style == "luxury":
                    price_range = (200, 500)
                else:
                    price_range = (100, 200)

                hotels = [
                    {"name": "Hotel G Singapore", "price": 120},
                    {"name": "Marina Bay Sands", "price": 300},
                    {"name": "Hotel Mono", "price": 95}
                ]

                filtered = [h for h in hotels if price_range[0] <= h["price"] <= price_range[1]]
                selected = filtered[0] if filtered else hotels[0]

                results["hotels"] = {
                    "name": selected["name"],
                    "rate_per_night": selected["price"],
                    "nights": nights,
                    "total": selected["price"] * nights
                }

            elif task == "search_restaurants_tripapi":
                results["restaurants"] = [
                    {"name": "Chilli Padi Nonya Cafe", "type": "local"},
                    {"name": "Din Tai Fung", "type": "chinese"},
                    {"name": "Candlenut", "type": "fine dining"}
                ]

            elif task == "find_attractions":
                prompt = f"List must-visit attractions in {query_data['destination']} for interests: {query_data['preferences']}"
                results["attractions"] = self.llm.call(prompt)

            elif task == "generate_itinerary":
                prompt = user_context + "\n\nGenerate a detailed multi-day itinerary based on this trip."
                results["itinerary"] = self.llm.call(prompt)

            elif task == "estimate_costs":
                prompt = user_context + "\n\nBreak down estimated trip costs by category."
                results["costs"] = self.llm.call(prompt)

        return results