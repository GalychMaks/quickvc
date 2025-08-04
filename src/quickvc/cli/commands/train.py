from pathlib import Path

from quickvc.training.train import Trainer


def train(config_path: Path = Path("configs/quickvc.json")) -> None:
    trainer = Trainer(config_path)
    trainer.train()
