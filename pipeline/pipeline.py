from io import BytesIO
from typing import Dict, List

from summary.summary import get_summary_service
from transcription.transcription import get_transcription_service


class DeepenPipelineRemote:
    def __init__(
        self,
        transcription_service,
        summary_service,
    ):
        self.transcription_service = transcription_service
        self.summary_service = summary_service

    def get_readable_transcript(self, turns: List | Dict) -> str:
        return self.transcription_service.get_readable_transcript(turns)

    def transcribe(self, audio_obj: BytesIO) -> str:
        return self.transcription_service.transcribe(audio_obj)

    def summarize(self, transcript: str) -> str:
        return self.summary_service.summarize(transcript=transcript)


def get_pipeline_service(config: Dict, metadata: Dict):
    transcription_service = get_transcription_service(config["transcription"])
    summary_service = get_summary_service(
        config["summarization"], metadata["youtube_url"]
    )
    return DeepenPipelineRemote(
        transcription_service=transcription_service,
        summary_service=summary_service,
    )
