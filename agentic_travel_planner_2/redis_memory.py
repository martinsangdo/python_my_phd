import redis
import json
from config import REDIS_URL

class MemoryModule:
    def __init__(self):
        self.client = redis.from_url(REDIS_URL)

    def save_preferences(self, user_id: str, prefs: dict):
        self.client.set(f"user:{user_id}:prefs", json.dumps(prefs))

    def get_preferences(self, user_id: str):
        raw = self.client.get(f"user:{user_id}:prefs")
        return json.loads(raw) if raw else {}