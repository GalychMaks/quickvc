import logging
from pathlib import Path

import click
import torch
from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)


def remove_silence(input_path: Path) -> AudioSegment:
    """
    Return a pydub.AudioSegment containing ONLY the concatenated speech.
    """

    # torch.set_num_threads(1)
    model, utils = torch.hub.load(
        "snakers4/silero-vad", "silero_vad", force_reload=False
    )
    get_ts, _, read_audio, _, _ = utils

    # read_audio → 16 kHz mono torch.Tensor
    wav = read_audio(str(input_path))
    speech_timestamps = get_ts(wav, model, return_seconds=True)

    if not speech_timestamps:
        logger.warning("No speech detected.")
        return AudioSegment.empty()

    full_audio = AudioSegment.from_file(input_path)
    speech_only = AudioSegment.empty()

    for timestamp in speech_timestamps:
        start_ms = int(timestamp["start"] * 1000)
        end_ms = int(timestamp["end"] * 1000)
        speech_only += full_audio[start_ms:end_ms]

    return speech_only


def split_chunks(
    audio: AudioSegment,
    output_dir: Path,
    chunk_length: int,
    normalize: bool,
    prefix: str,
    fmt: str,
):
    """Split `audio` into exact chunk_length-second pieces, export to output_dir."""
    chunk_ms = chunk_length * 1000
    total_ms = len(audio)
    num_chunks = total_ms // chunk_ms

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Splitting into {num_chunks} chunks of {chunk_length}s each…")

    for i in range(num_chunks):
        start = i * chunk_ms
        end = start + chunk_ms
        chunk = audio[start:end]
        if normalize:
            chunk = effects.normalize(chunk)
        out_path = output_dir / f"{prefix}_{i + 1}.{fmt}"
        chunk.export(out_path, format=fmt)
        logger.debug(f"Exported {out_path.name}")


@click.command()
@click.option(
    "--input-file", "-i", type=click.Path(exists=True, path_type=Path), required=True
)
@click.option("--output-dir", "-o", type=click.Path(path_type=Path), required=True)
@click.option(
    "--chunk-length",
    "-l",
    type=int,
    default=20,
    show_default=True,
    help="Seconds per chunk.",
)
@click.option("--normalize", is_flag=True, help="Normalize each chunk.")
@click.option("--prefix", default="chunk", show_default=True, help="Filename prefix.")
@click.option(
    "--output-format",
    default="mp3",
    show_default=True,
    type=click.Choice(["mp3", "wav", "flac", "ogg", "aac"], case_sensitive=False),
)
def split(input_file, output_dir, chunk_length, normalize, prefix, output_format):
    """
    1) Remove silence via Silero VAD → one long speech-only AudioSegment.
    2) Split that segment into fixed-length chunks (no silence) and export.
    """
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
    )

    speech_audio = remove_silence(input_file)
    if len(speech_audio) < chunk_length * 1000:
        logger.warning("Not enough speech to form even one chunk.")
        return

    split_chunks(
        speech_audio, output_dir, chunk_length, normalize, prefix, output_format
    )
    logger.info("All done.")


if __name__ == "__main__":
    logging.basicConfig(
        level=logging.DEBUG,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )
    split()
