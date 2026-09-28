import json
import time
import urllib.request

URL = "http://localhost:11434/api/generate"
MODEL = "mistral:7b-instruct-q4_K_M"

prompt = """
Explain how a large language model inference system is deployed and operated in a production environment. Start with the trained model artifact and describe how the model, tokenizer, dependencies, and runtime configuration are packaged into a reproducible environment. Explain how Docker containers help ensure that the same software stack runs consistently across development, testing, and production systems. Describe how an inference server exposes an API that accepts prompts, validates requests, tokenizes input, executes the model, generates tokens, and returns responses to clients.

Discuss the main performance metrics used for LLM inference. Explain time to first token, token generation speed, latency, throughput, and why each metric provides different information about system performance. Describe the difference between the prefill phase and the decode phase. Explain why processing a longer input prompt can increase time to first token, while token generation speed during decoding may remain relatively stable. Discuss how CPU and GPU hardware affect inference performance and why quantization can reduce memory requirements and computational cost.

Explain how production inference systems handle multiple users. Discuss concurrency, request queues, batching, resource limits, and load balancing. Describe why CPU inference can become a bottleneck when several users submit requests simultaneously. Explain how continuous batching on GPU-based inference servers can improve aggregate throughput by processing tokens from multiple requests efficiently.

Describe the observability requirements of an inference service. Explain how engineers can monitor request counts, response latency, time to first token, tokens per second, CPU usage, memory consumption, error rates, and queue depth. Discuss why logs are important for troubleshooting failed requests and provider errors. Explain how alerts can notify operators when latency or error rates exceed acceptable thresholds.

Discuss reliability and fault tolerance for LLM applications. Explain how a gateway can route requests between different inference providers. Describe a system where a local model is the primary provider and a remote API is configured as a fallback. Explain what should happen when the local provider becomes unavailable and how automatic fallback improves service availability. Discuss timeouts, retries, rate limits, and circuit breakers as techniques for preventing failures from propagating through the system.

Finally, explain why load testing is necessary before deploying an inference service to production. Describe how tools such as Locust can simulate concurrent users and measure P50, P95, and P99 latency. Explain how engineers can use these results to identify the point where the system becomes saturated. Conclude by describing how benchmarking, monitoring, fallback providers, load testing, and infrastructure optimization work together to create a reliable and efficient LLM inference platform.
"""

payload = json.dumps({
    "model": MODEL,
    "prompt": prompt,
    "stream": True,
    "options": {
        "num_predict": 50
    }
}).encode()

request = urllib.request.Request(
    URL,
    data=payload,
    headers={"Content-Type": "application/json"}
)

start = time.perf_counter()
first_token_time = None
final_data = None

with urllib.request.urlopen(request) as response:
    for line in response:
        data = json.loads(line)

        if data.get("response") and first_token_time is None:
            first_token_time = time.perf_counter()

        if data.get("done"):
            final_data = data

ttft_ms = (first_token_time - start) * 1000
tps = final_data["eval_count"] / (final_data["eval_duration"] / 1e9)

print("Prompt tokens:", final_data["prompt_eval_count"])
print(f"TTFT: {ttft_ms:.2f} ms")
print(f"TPS: {tps:.2f}")
