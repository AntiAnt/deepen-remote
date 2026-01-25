from typing import Dict

import requests


class LlamaMSummaryService:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        prompt = f"""You are an expert at creating profound, beautiful summaries.

        Write a concise, moving summary in clean markdown with exactly these sections:

        1. Core Message (1 sentence)
        2. Key Insights (4–6 bullets)
        3. Most Powerful Moment
        4. Final Takeaway

        Transcript:
        {transcript}

        Summary in clean markdown:"""

        print("Generating summary with llama3.1:8b via API (clean, fast, perfect)…")

        response = requests.post(
            "http://localhost:11434/api/generate",
            json={
                "model": self.model,
                "prompt": prompt,
                "stream": False,
                "options": {
                    "temperature": 0.7,
                    "num_ctx": 2048,
                },
            },
            timeout=300,
        )

        if response.status_code != 200:
            print(f"Error: {response.status_code}")
            print(response.text)
            exit(1)

        result = response.json()
        summary = result["response"].strip()

        return summary


def get_summary_service(config: Dict):
    print(f"SUMMARY CONFIG: {config}")
    return LlamaMSummaryService(model=config["models"]["llama"])
