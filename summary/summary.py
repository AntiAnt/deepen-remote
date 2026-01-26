import os
from typing import Dict
from xai_sdk import Client
from xai_sdk.chat import user, system

import requests

PROMPT = """
Task: Transform this interview transcript into a deep learning artifact that turns passive listening into active expertise acquisition.

Required structure (do not deviate):

1. Title
   Robust Summary and Deepened Analysis of the Interview: [interviewer] interviewing [guest] – [main topic in 8 words or fewer]

2. Core Summary of the Interview (~600–900 words)
   - Faithful but concise retelling of the conversation arc
   - Highlight all major conceptual distinctions, metaphors, personal anecdotes, neuroscientific claims, biblical/philosophical references

3. Deepening Understanding (~800–1200 words)
   - Select and explain in depth the 5–7 most intellectually important ideas
   - For each idea give:
     a. Clear restatement
     b. Underlying mechanisms (neuroscience, psychology, evolution, philosophy)
     c. Broader context or adjacent research (even if not mentioned)
     d. Concrete, non-obvious real-world applications

4. Insights and Relationships Not Mentioned (~400–700 words)
   - Original connections across disciplines
   - Unstated societal / developmental / leadership implications
   - Possible critiques, edge cases or counter-evidence
   - Links to thinkers / frameworks that illuminate the discussion indirectly

5. List of Sources Discussed
   - Every proper noun reference to books, papers, experiments, historical figures, biblical stories etc.
   - Format: Author (Year if given). Title. Brief one-line what it contributed.

6. Relevant Sources Not Mentioned
   - 10–15 carefully chosen books / papers / authors
   - Each with title, author, year (approx if needed), and 1–2 sentences explaining exactly how it builds on or contrasts with ideas in the transcript

Rules:
- Total length 2200–3000 words
- Maximize density and insight per word
- Never fabricate facts; if connecting externally, make the logical bridge transparent
- Write for an intelligent, curious adult who wants to master these topics over time

Transcript follows:

"""


class LlamaMSummaryService:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        prompt = "\n".join([PROMPT, transcript])

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


class XAISummaryService:
    def __init__(self, model: str) -> None:
        self.model = model

    def summarize(self, transcript: str) -> str:
        client = Client(api_key=os.getenv("XAI_API_KEY"), timeout=3600)

        chat = client.chat.create(model=self.model)
        chat.append(system("""
            You are expert AI educator specializing in breaking down and teaching about various media formats like podcasts, transcripts, videos, articles, or interviews. Your goal is to help users gain deeper understanding and expertise by transforming passive consumption into active learning.

            Key guidelines:
            - Always start by summarizing the media's core content faithfully and concisely.
            - Highlight key themes, concepts, anecdotes, references, and distinctions (e.g., metaphors, historical context).
            - Deepen understanding: Select 4-6 important ideas, explain mechanisms (e.g., psychological, cultural), provide broader context/research, and suggest real-world applications.
            - Offer original insights: Connect to other disciplines, implications, critiques, and related thinkers.
            - End with interactive elements: Pose questions, exercises, or follow-up prompts to encourage user engagement and mastery.
            - Keep responses structured, insightful, and accessible for curious adults. Use bullet points, numbered lists, or tables for clarity.
            - If the media is specialized (e.g., music, science), draw on relevant knowledge without fabricating facts.
            - Be encouraging, neutral, and fun – make learning enjoyable!
            """))
        chat.append(user(f"""Analyze this transcript and teach me about it:
            {transcript}
            """))
        response = chat.sample()
        return response.content


def get_summary_service(config: Dict):
    if config["type"] == "llama":
        return LlamaMSummaryService(model=config["models"]["llama"])
    if config["type"] == "xai":
        return XAISummaryService(model=config["models"]["xai"])
