# class is used to get the names of all avilable models in the SAIA API. Not really used in the project.
from openai import OpenAI
import os
from dotenv import load_dotenv

load_dotenv()

client = OpenAI(
    api_key=os.getenv("SAIA_API_KEY"),
    base_url=os.getenv("SAIA_BASE_URL"),
    timeout=60
)

payload = client.get("/models", cast_to=object)
models = payload.get("data", payload) if isinstance(payload, dict) else payload

for m in models:
    print(m["id"])