import argparse

from reliquery import Relic

from pipeline.pipeline import get_pipeline_service


def main():
    parser = argparse.ArgumentParser(
        description="Deepen pipeline: ingest, transcribe, and summarize informational videos.",
        formatter_class=argparse.RawTextHelpFormatter,
    )

    parser.add_argument("--relic-name", required=True)
    parser.add_argument("--relic-type", required=True)
    parser.add_argument("--storage-name", required=True)

    parser.add_argument(
        "--steps",
        nargs="+",  # allows multiple values, e.g. --steps ingest transcribe
        default=["ingest", "transcribe", "summarize"],  # run all by default
        choices=["ingest", "transcribe", "summarize"],  # validate inputs
        help="Pipeline steps to run (default: all). Order matters. Examples:\n"
        "  --steps ingest\n"
        "  --steps ingest transcribe\n"
        "  --steps summarize (skips previous)",
    )

    args = parser.parse_args()
    relic_name = args.relic_name
    relic_type = args.relic_type
    storage_name = args.storage_name
    import os

    assert os.path.exists("/home/ubuntu/reliquery")
    assert os.path.exists("/home/ubuntu/reliquery/config")
    assert os.path.exists("~/reliquery/config")




    relic = Relic(name=relic_name, relic_type=relic_type, storage_name=storage_name)

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
