class MemoryStore:
    def __init__(self):
        self.store_data = {}

    def store(self, key, value):
        self.store_data[key] = value

    def recall(self, key):
        return self.store_data.get(key)