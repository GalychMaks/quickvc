import logging
from pathlib import Path
from typing import cast

import torch
import typer
from pydub import AudioSegment, effects

logger = logging.getLogger(__name__)
split_typer = typer.Typer()


def remove_silence(input_path: Path) -> AudioSegment:
    """
    Return a pydub.AudioSegment containing ONLY the concatenated speech.
    """

    # torch.set_num_threads(1)
    model, utils = torch.hub.load(  # type: ignore
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


def split(
    input_file: Path,
    output_dir: Path,
    chunk_length: int = 20,
    normalize: bool = False,
    prefix: str = "chunk",
    output_format: str = "mp3",
) -> None:
    """
    1) Remove silence via Silero VAD → one long speech-only AudioSegment.
    2) Split that segment into fixed-length chunks (no silence) and export.
    """

    if not input_file.exists() or not input_file.is_file():
        raise typer.BadParameter(f"Input file does not exist: {input_file}")

    # Create output directory if needed
    output_dir.mkdir(parents=True, exist_ok=True)

    # Do the work
    speech_audio = remove_silence(input_file)
    if len(speech_audio) < chunk_length * 1000:
        logger.warning("Not enough speech to form even one chunk.")
        raise typer.Exit(code=0)

    chunk_ms = chunk_length * 1000
    total_ms = len(speech_audio)
    num_chunks = total_ms // chunk_ms

    output_dir.mkdir(parents=True, exist_ok=True)
    logger.info(f"Splitting into {num_chunks} chunks of {chunk_length}s each…")

    for i in range(num_chunks):
        start = i * chunk_ms
        end = start + chunk_ms
        chunk = cast(AudioSegment, speech_audio[start:end])
        if normalize:
            chunk = effects.normalize(chunk)
        out_path = output_dir / f"{prefix}_{i + 1}.{output_format}"
        chunk.export(out_path, format=output_format)
        logger.debug(f"Exported {out_path.name}")

    logger.info("All done.")
