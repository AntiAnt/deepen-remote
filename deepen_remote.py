import argparse
import os

from reliquery import Relic

from pipeline.pipeline import get_pipeline_service


def main():
    parser = argparse.ArgumentParser(
        description="Deepen pipeline: ingest, transcribe, and summarize informational videos.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--relic-name", required=True)
    parser.add_argument("--relic-type", required=True)
    parser.add_argument("--reliquery-config-root", required=True)

    args = parser.parse_args()
    relic_name = args.relic_name
    relic_type = args.relic_type
    reliquery_config_root = args.reliquery_config_root

    relic = Relic(
        name=relic_name,
        relic_type=relic_type,
        storage_name="remote_s3",
        reliquery_config_root=reliquery_config_root,
    )
  
    assert Relic.relic_exists(name=relic_name, relic_type=relic_type, storage_name="remote_s3")
    relic.add_text(name="remote-test", text="confirmign I can write to bucket")
    remote_config = relic.get_json(name="remote-config")

    pipeline = get_pipeline_service(remote_config)
    # transcription

    transcription = pipeline.transcribe(relic.get_audio(name="audio.wav"))
    readable_transcript = pipeline.get_readable_transcript(transcription)

    relic.add_json(name="whisper-transcript", json_data=transcription)
    relic.add_text(name="readable-whisper-transcript", text=readable_transcript)

    # Summary
    summary = pipeline.summarize(transcript=readable_transcript)

    relic.add_text(name="summary-llm", text=summary)

    print("Main: audio processing complete.")


if __name__ == "__main__":
    main()
