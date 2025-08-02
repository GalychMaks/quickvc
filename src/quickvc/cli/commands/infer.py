import logging
import os
import time
from pathlib import Path

import soundfile as sf
import torch

from quickvc.wrapper import QuickVC

logger = logging.getLogger(__name__)


def infer(
    config: Path = Path("checkpoints/pretrained/config.yaml"),
    checkpoint: Path = Path("checkpoints/pretrained/G_1200000.pth"),
    source: Path = Path("data/source.wav"),
    target: Path = Path("data/target.wav"),
    outdir: Path = Path("output/quickvc"),
) -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"

    logger.info("Initializing QuickVC")
    quickvc = QuickVC(
        config_path=config,
        checkpoint_path=checkpoint,
        device=device,
    )

    os.makedirs(outdir, exist_ok=True)

    logger.info("Synthesizing...")
    generated_np, sr = quickvc.convert_voice(source, target)

    logger.info("Saving generated audio...")
    timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
    output_path = Path(outdir) / f"generated_{timestamp}.wav"
    sf.write(output_path, generated_np, sr, format="wav")
    logger.info(f"Saved to '{output_path}'")
