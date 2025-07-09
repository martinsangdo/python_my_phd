class AdaptiveReasoner:
    def reflect_and_adapt(self, tool_outputs: dict, query_data: dict) -> dict:
        reflections = []

        if "costs" in tool_outputs:
            try:
                cost_info = tool_outputs["costs"]
                total_cost = float(cost_info.get("total_estimated_cost", 0))
                if total_cost > query_data.get("budget", float("inf")):
                    reflections.append("⚠️ Estimated cost exceeds budget. Consider adjusting accommodation or flights.")
            except:
                reflections.append("⚠️ Unable to parse cost total for reflection.")

        if "itinerary" in tool_outputs:
            itinerary_lines = tool_outputs["itinerary"].splitlines()
            days_with_many_items = [line for line in itinerary_lines if line.count("•") > 4]
            if len(days_with_many_items) > 1:
                reflections.append("⚠️ Some days in the itinerary may be too packed. Consider spacing out activities.")

        if "flights" in tool_outputs and "alternative" in tool_outputs["flights"]:
            reflections.append("💡 A cheaper flight alternative is available.")

        if reflections:
            tool_outputs["reflections"] = reflections

        return tool_outputs