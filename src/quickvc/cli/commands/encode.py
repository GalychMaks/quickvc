import numpy as np
from pathlib import Path
from tqdm import tqdm

import torch
import torchaudio
from torchaudio.functional import resample


def encode_dataset(dataset_dir: Path = Path("data/chunks")):
    device = "cuda" if torch.cuda.is_available() else "cpu"

    print("Loading hubert checkpoint")
    hubert = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device).eval()  # type: ignore

    print(f"Encoding dataset at {dataset_dir}")
    for in_path in tqdm(list(Path(dataset_dir).rglob("*wav"))):
        out_path = Path(dataset_dir) / in_path.relative_to(dataset_dir)

        if True:  # not os.path.exists(out_path.with_suffix(".npy")):
            wav, sr = torchaudio.load(in_path)
            wav = resample(wav, sr, 16000)
            wav = wav.unsqueeze(0).to(device)

            with torch.inference_mode():
                units = hubert.units(wav)

            np.save(out_path.with_suffix(".npy"), units.squeeze().cpu().numpy())


if __name__ == "__main__":
    encode_dataset()
