from typing import Dict

import requests


class LlamaMSummaryService:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        # prompt = f"""You are an expert at creating profound, beautiful summaries.

        # Write a concise, moving summary in clean markdown with exactly these sections:

        # 1. Core Message (1 sentence)
        # 2. Key Insights (4–6 bullets)
        # 3. Most Powerful Moment
        # 4. Final Takeaway

        # Transcript:
        # {transcript}

        # Summary in clean markdown:"""

        prompt = f"""You are an expert at creating insightful, comprehensive summaries that not only capture the essence but also deepen understanding and provide pathways for further learning.

        Analyze the transcript and produce output in clean markdown with exactly these sections:

        1. Core Message  (paragraph summarizing the overall theme)
        2. Detailed Key Insights (list of bullets covering main points, ideas, and discussions with sufficient detail to convey a decent amount of information from the transcript)
        3. Deeper Dive into Subjects (For 3-4 main subjects discussed, provide a brief paragraph each going slightly deeper, explaining concepts, implications, or related ideas beyond what's in the transcript)
        4. Relevant Sources for Further Learning (3-5 bullets listing key subjects with suggested reliable sources like books, articles, websites, or experts, including why they're useful)

        Ensure the output is engaging, objective, intellectual, and encourages curiosity.

        Aim for 10 to 15 minute read.

        Transcript:
        {transcript}

        Output in clean markdown:"""

        print(f"Generating summary with {self.model}")

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
