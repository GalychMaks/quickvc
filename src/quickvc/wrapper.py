import logging
from pathlib import Path
from typing import Optional
import librosa
import numpy as np
from omegaconf import OmegaConf
import torch

from quickvc.modules.models import SynthesizerTrn
from quickvc.utils.mel_processing import mel_spectrogram_torch
from quickvc.utils.utils import load_audio_array

logger = logging.getLogger(__name__)


class QuickVC:
    def __init__(
        self,
        config_path: Path | str,
        checkpoint_path: Path | str,
        device: Optional[str] = None,
    ):
        # Set device
        if device is None:
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)
        logger.info(f"Torch device used: {self.device}")

        logger.info(f"Loading config from '{config_path}'")
        self.config = OmegaConf.load(config_path)

        self.sr = self.config.data.sampling_rate
        self.mel_fn = lambda x: mel_spectrogram_torch(
            x,
            self.config.data.filter_length,
            self.config.data.n_mel_channels,
            self.config.data.sampling_rate,
            self.config.data.hop_length,
            self.config.data.win_length,
            self.config.data.mel_fmin,
            self.config.data.mel_fmax,
        )

        logger.info("Loading hubert model")
        self.hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(
            self.device
        )

        self.net_g = SynthesizerTrn(
            self.config.data.filter_length // 2 + 1,
            self.config.train.segment_size // self.config.data.hop_length,
            **self.config.model,
        ).to(device)

        logger.info("Loading quickvc model...")
        self.load_checkpoint(checkpoint_path)

    def load_checkpoint(self, checkpoint_path: Path | str):
        assert Path(checkpoint_path).exists()

        checkpoint_dict = torch.load(checkpoint_path, map_location="cpu")

        saved_state_dict = checkpoint_dict["model"]

        # Retrieve current state
        if hasattr(self.net_g, "module"):  # DDP
            state_dict = self.net_g.module.state_dict()
        else:
            state_dict = self.net_g.state_dict()

        # Load weights from checkpoint
        new_state_dict = {}
        for k, v in state_dict.items():
            try:
                new_state_dict[k] = saved_state_dict[k]
            except Exception:
                logger.info("%s is not in the checkpoint" % k)
                new_state_dict[k] = v

        # Update current state with loaded weights
        if hasattr(self.net_g, "module"):  # DDP
            self.net_g.module.load_state_dict(new_state_dict)
        else:
            self.net_g.load_state_dict(new_state_dict)

        logger.info(f"Loaded checkpoint: '{checkpoint_path}'")

    @torch.inference_mode()
    def convert_voice(
        self,
        source: tuple[np.ndarray, int] | Path | str,
        target: tuple[np.ndarray, int] | Path | str,
    ) -> tuple[np.ndarray, int]:
        # Load target
        target_np = load_audio_array(target, sr=self.sr)
        target_np, _ = librosa.effects.trim(target_np, top_db=20)
        target_tensor = torch.from_numpy(target_np).unsqueeze(0).to(self.device)
        target_mel = self.mel_fn(target_tensor)

        # Load Source
        source_np = load_audio_array(source, sr=self.sr)
        source_tensor = (
            torch.from_numpy(source_np).unsqueeze(0).unsqueeze(0).to(self.device)
        )

        # Extract speech features
        hubert_units = self.hubert.units(source_tensor).transpose(2, 1)

        # Run inference
        audio = self.net_g.infer(hubert_units, mel=target_mel)

        return audio[0][0].data.cpu().float().numpy(), self.sr
