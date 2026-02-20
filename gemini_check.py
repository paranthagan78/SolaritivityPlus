# debug_gemini.py  — run this from your project root: python debug_gemini.py
import os
from dotenv import load_dotenv
load_dotenv()

import requests

GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY") or os.environ.get("GOOGLE_API_KEY")
print(f"Key loaded: {'YES, ends with ...' + GEMINI_API_KEY[-6:] if GEMINI_API_KEY else 'NO KEY FOUND'}")

# Step 1: List models
url = f"https://generativelanguage.googleapis.com/v1beta/models?key={GEMINI_API_KEY}"
resp = requests.get(url, timeout=30)
print(f"\nList models status: {resp.status_code}")
print(resp.text[:2000])