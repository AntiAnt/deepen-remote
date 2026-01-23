import argparse
import os

from reliquery import Relic

from pipeline.pipeline import get_pipeline_service


def print_directory_tree(root_path: str, prefix: str = "", level: int = 0):
    """
    Recursively print a simple file tree of the given directory.
    """
    if not os.path.isdir(root_path):
        print(f"{prefix} (not a directory or does not exist)")
        return

    try:
        items = sorted(os.listdir(root_path))
    except PermissionError:
        print(f"{prefix}Permission denied")
        return
    except Exception as e:
        print(f"{prefix}Error listing directory: {e}")
        return

    for i, item in enumerate(items):
        path = os.path.join(root_path, item)
        is_last = i == len(items) - 1

        # Tree branch symbol
        branch = "└── " if is_last else "├── "
        print(f"{prefix}{branch}{item}")

        # Recurse into directories
        if os.path.isdir(path):
            new_prefix = prefix + ("    " if is_last else "│   ")
            print_directory_tree(path, new_prefix, level + 1)


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

    try:
        relic = Relic(
            name=relic_name,
            relic_type=relic_type,
            storage_name="remote_s3",
            reliquery_config_root=reliquery_config_root,
        )
    except Exception as e:
        print(f"Failed to create Relic: {type(e).__name__}: {str(e)}")
        print(f"Root path used: {reliquery_config_root}")

        # Print file tree of the root directory
        print("\nFile tree of root directory:")
        print_directory_tree(reliquery_config_root)

        # Optional: re-raise or return None depending on your needs
        raise RuntimeError

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
