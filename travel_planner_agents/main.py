from agents.planner_agent import PlannerAgent

if __name__ == "__main__":
    planner = PlannerAgent()
    user_input = input("Where would you like to go? ")
    plan = planner.run(user_input)
    print("\nFinal Trip Plan:\n")
    print(plan)