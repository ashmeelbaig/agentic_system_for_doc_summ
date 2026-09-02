from dotenv import load_dotenv
from huggingface_hub import InferenceClient
import os

load_dotenv()

HF_TOKEN = os.getenv("HF_TOKEN")

MODELS_TO_TEST = [
    "openai/gpt-oss-20b:groq",
    "openai/gpt-oss-120b:groq",
    "google/gemma-2-2b-it",
    "Qwen/Qwen3-4B-Thinking-2507",
    "Qwen/Qwen2.5-7B-Instruct-1M",
    "Qwen/Qwen2.5-Coder-32B-Instruct",
    "meta-llama/Llama-3.1-8B-Instruct:nscale",
    "meta-llama/Llama-3.1-8B-Instruct:ovhcloud",
    "meta-llama/Llama-3.1-8B-Instruct:scaleway",
]

PROMPT = "Answer in one sentence: What is Retrieval Augmented Generation?"

if not HF_TOKEN:
    raise RuntimeError("HF_TOKEN is missing. Add it to your .env file.")

for model_id in MODELS_TO_TEST:
    print("\n" + "=" * 80)
    print(f"Testing: {model_id}")
    print("=" * 80)

    try:
        client = InferenceClient(
            token=HF_TOKEN,
            timeout=120,
        )

        response = client.chat_completion(
            model=model_id,
            messages=[
                {
                    "role": "user",
                    "content": PROMPT,
                }
            ],
            max_tokens=100,
            temperature=0.1,
        )

        answer = response.choices[0].message.content
        print("STATUS: SUCCESS")
        print("ANSWER:", answer)

    except Exception as e:
        print("STATUS: FAILED")
        print("ERROR TYPE:", type(e).__name__)
        print("ERROR:", str(e)[:1000])