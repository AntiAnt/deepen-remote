import json
import tempfile
import wave
from datetime import datetime
from io import BytesIO
from pathlib import Path
from typing import Dict, List
from faster_whisper import WhisperModel
from pyannote.audio import Pipeline
from reliquery import Relic
from vosk import KaldiRecognizer, Model


class VoskTranscriber:
    def __init__(
        self,
        vosk_model: str,
        pyannote_model: str,
        hf_token: str,
    ):
        self.vosk_model = Model(vosk_model)
        self.diarization = Pipeline.from_pretrained(pyannote_model, token=hf_token)

    def get_readable_transcript(
        self,
        turns: List[Dict],
        speaker_map={"SPEAKER_00": "host", "SPEAKER_01": "guest"},
    ) -> str:
        """
        Generates a human readable version of a Vosk generated transcription.
        """
        text_lines = []
        text_lines.append("=" * 60 + "\n")
        text_lines.append("Human Friendly Transcription\n")
        text_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        text_lines.append("=" * 60 + "\n\n")

        for turn in turns:
            speaker_id = turn["speaker"]
            speaker = speaker_map.get(speaker_id, speaker_id)
            text = turn["text"].strip()
            start = turn.get("start", 0.0)

            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"

            text_lines.append(f"{timestamp} {speaker}")
            wrapped = "\n".join(text[i : i + 80] for i in range(0, len(text), 80))
            text_lines.append(f"\t{wrapped}\n\n")

        readable_text = "".join(text_lines)
        print(readable_text)
        return readable_text

    def transcribe(self, audio: BytesIO) -> List[Dict]:
        """
        Transcribe audio_bytes using Vosk + pyannote diarization.
        Returns list of turns: [{"speaker": "SPEAKER_00", "text": "...", "start": float, "end": float}, ...]
        No temp files created.
        """
        print(
            "Transcription: Running Vosk transcription + pyannote diarization (in-memory)..."
        )
        audio.seek(0)
        diarization = self.diarization(audio)

        audio.seek(0)
        with wave.open(audio, "rb") as wf:
            rec = KaldiRecognizer(self.vosk_model, wf.getframerate())
            rec.SetWords(True)

            words = []
            while True:
                data = wf.readframes(4000)
                if len(data) == 0:
                    break
                if rec.AcceptWaveform(data):
                    part = json.loads(rec.Result())
                    words.extend(part.get("result", []))

            final = json.loads(rec.FinalResult())
            words.extend(final.get("result", []))

        turns = []
        current_speaker = None
        current_text = []
        current_start = None

        for word in words:
            start = word["start"]
            text = word["word"]
            speaker = next(
                (
                    spk
                    for turn, _, spk in diarization.itertracks(yield_label=True)
                    if turn.start <= start <= turn.end
                ),
                "UNKNOWN",
            )

            if speaker != current_speaker and current_text:
                turns.append(
                    {
                        "speaker": current_speaker,
                        "text": " ".join(current_text),
                        "start": current_start,
                        "end": start,
                    }
                )
                current_text = []

            if not current_text:
                current_start = start
            current_text.append(text)
            current_speaker = speaker

        # Final turn
        if current_text and words:
            turns.append(
                {
                    "speaker": current_speaker,
                    "text": " ".join(current_text),
                    "start": current_start,
                    "end": words[-1]["end"],
                }
            )

        print(f"Transcription: Done - {len(turns)} speaker turns")

        return turns


class WhisperTranscriber:
    def __init__(self, config: dict):
        whisper_config = config.get("models", {}).get("whisper", {})
        self.model_size = whisper_config.get("model_size", "medium")
        self.device = config.get("device", "cpu")  # Change in config for EC2/M1
        self.compute_type = config.get("compute_type", "float16")
        self.language = config.get("language", "en")
        self.beam_size = config.get("beam_size", 5)

        print(f"[WhisperTranscriber] Loading model: {self.model_size} on {self.device}")

        # Load the model
        self.model = WhisperModel(
            self.model_size, device=self.device, compute_type=self.compute_type
        )

    def get_readable_transcript(self, turns: List[Dict]) -> str:
        """
        Generates a human readable version of a Whisper generated transcription.
        """
        text_lines = []
        text_lines.append("=" * 60 + "\n")
        text_lines.append("Human Friendly Transcription\n")
        text_lines.append(
            f"Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n"
        )
        text_lines.append("=" * 60 + "\n\n")

        for turn in turns:
            start = turn.get("start", 0.0)
            minutes = int(start // 60)
            seconds = int(start % 60)
            timestamp = f"[{minutes:02d}:{seconds:02d}]"
            text = turn["text"]

            text_lines.append(timestamp)
            wrapped = "\n".join(text[i : i + 80] for i in range(0, len(text), 80))
            text_lines.append(f"\t{wrapped}\n\n")

        return "".join(text_lines)

    def transcribe(self, audio: BytesIO) -> List:
        """
        Transcribe audio to list of segments.
        Returns: [{'start': float, 'end': float, 'text': str}]
        """
        segments, info = self.model.transcribe(
            audio=audio,
            beam_size=self.beam_size,
            language=self.language,
            vad_filter=True,  # Removes silence
        )

        result = []
        for segment in segments:
            result.append(
                {
                    "start": segment.start,
                    "end": segment.end,
                    "text": segment.text.strip(),
                }
            )

        print(
            f"[Whisper] Detected language: {info.language} (prob: {info.language_probability:.2f})"
        )
        return result


def get_transcription_service(config: Dict):
    if config["type"] == "whisper":
        return WhisperTranscriber(config=config)
    elif config["type"] == "vosk":
        return VoskTranscriber(
            vosk_model=config["models"]["vosk"],
            pyannote_model=config["models"]["pyannote"],
            hf_token=config["secrets"]["hf_token"],
        )
    else:
        raise RuntimeError("transcriber config missing or not supported.")
