class BudgetAgent:
    def __init__(self, memory):
        self.memory = memory

    def evaluate(self, flight, hotel, activities):
        print("[BudgetAgent] Calculating total cost...")
        return flight["price"] + hotel["price"] + activities["total_price"]