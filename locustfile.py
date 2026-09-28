from locust import HttpUser, task, between
import random

PROMPTS = [
    "Explain what Docker is in two sentences.",
    "What is Kubernetes? Answer briefly.",
    "Explain the difference between CPU and GPU.",
    "What is MLOps? Answer in two sentences.",
]


class LLMUser(HttpUser):
    wait_time = between(1, 2)

    @task
    def chat(self):
        prompt = random.choice(PROMPTS)

        self.client.post(
            "/v1/chat/completions",
            json={
                "model": "chat",
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "max_tokens": 50
            },
            timeout=120
        )
