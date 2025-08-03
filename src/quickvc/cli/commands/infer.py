import logging
import time
from pathlib import Path
from typing import Optional

import soundfile as sf
import torch

from quickvc.utils.timer import Timer
from quickvc.wrapper import QuickVC

logger = logging.getLogger(__name__)


def infer(
    config_path: Path = Path("checkpoints/pretrained/config.yaml"),
    checkpoint_path: Path = Path("checkpoints/pretrained/G_1200000.pth"),
    source: Path = Path("data/source.wav"),
    target: Path = Path("data/target.wav"),
    output_dir: Path = Path("output/quickvc"),
    output_name: Optional[str] = None,
) -> None:
    # Initialize
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Initializing QuickVC")
    quickvc = QuickVC(
        config_path=config_path,
        checkpoint_path=checkpoint_path,
        device=device,
    )

    # Run inferece
    logger.info("Synthesizing...")
    with Timer() as t:
        generated_np, sr = quickvc.convert_voice(source, target)

    # Compute RTF
    audio_duration = len(generated_np) / sr
    rtf = t.elapsed / audio_duration

    logger.info(
        f"Elapsed time: {t.elapsed:.4f}s, Audio duration: {audio_duration:.2f}s, RTF: {rtf:.4f} iRTF: {1 / rtf:.2f}"
    )

    # Save generated audio
    logger.info("Saving generated audio...")
    Path(output_dir).mkdir(parents=True, exist_ok=True)

    if output_name:
        output_path = Path(output_dir) / output_name
    else:
        timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
        output_path = Path(output_dir) / f"generated_{timestamp}.wav"

    sf.write(output_path, generated_np, sr, format="wav")
    logger.info(f"Saved to '{output_path}'")
