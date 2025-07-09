class ActionExecutor:
    def deliver_final_plan(self, plan: dict):
        print("🧭 Final Trip Plan")
        for key, value in plan.items():
            print(f"\n--- {key.upper()} ---\n{value}\n")