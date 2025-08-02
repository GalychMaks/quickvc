import argparse
import logging
import os
import time

import librosa
import torch
from scipy.io.wavfile import write

import quickvc.utils.utils as utils
from quickvc.modules.models import SynthesizerTrn
from quickvc.utils.mel_processing import mel_spectrogram_torch

logging.getLogger("numba").setLevel(logging.WARNING)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--hpfile",
        type=str,
        default="checkpoints/pretrained/config.json",
        help="path to json config file",
    )
    parser.add_argument(
        "--ptfile",
        type=str,
        default="checkpoints/pretrained/G_1200000.pth",
        help="path to pth file",
    )
    parser.add_argument(
        "--source",
        type=str,
        default="data/source.wav",
        help="path to source audio file",
    )
    parser.add_argument(
        "--target",
        type=str,
        default="data/target.wav",
        help="path to target audio file",
    )
    parser.add_argument(
        "--outdir", type=str, default="output/quickvc", help="path to output dir"
    )
    args = parser.parse_args()

    device = "cuda" if torch.cuda.is_available() else "cpu"

    os.makedirs(args.outdir, exist_ok=True)
    hps = utils.get_hparams_from_file(args.hpfile)

    print("Loading model...")
    net_g = SynthesizerTrn(
        hps.data.filter_length // 2 + 1,
        hps.train.segment_size // hps.data.hop_length,
        **hps.model,
    ).to(device)
    _ = net_g.eval()
    total = sum([param.nelement() for param in net_g.parameters()])

    print("Number of parameter: %.2fM" % (total / 1e6))
    print("Loading checkpoint...")
    _ = utils.load_checkpoint(args.ptfile, net_g, None)

    print("Loading hubert_soft checkpoint")
    hubert_soft = torch.hub.load("bshall/hubert:main", "hubert_soft").to(device)
    print("Loaded soft hubert.")

    print("Synthesizing...")

    with torch.no_grad():
        # target
        wav_tgt, _ = librosa.load(args.target, sr=hps.data.sampling_rate)
        wav_tgt, _ = librosa.effects.trim(wav_tgt, top_db=20)
        wav_tgt = torch.from_numpy(wav_tgt).unsqueeze(0).to(device)
        mel_tgt = mel_spectrogram_torch(
            wav_tgt,
            hps.data.filter_length,
            hps.data.n_mel_channels,
            hps.data.sampling_rate,
            hps.data.hop_length,
            hps.data.win_length,
            hps.data.mel_fmin,
            hps.data.mel_fmax,
        )
        # source
        wav_src, _ = librosa.load(args.source, sr=hps.data.sampling_rate)
        wav_src = torch.from_numpy(wav_src).unsqueeze(0).unsqueeze(0).to(device)
        print(wav_src.size())
        # long running
        # do something other
        c = hubert_soft.units(wav_src)

        c = c.transpose(2, 1)
        # print(c.size())
        audio = net_g.infer(c, mel=mel_tgt)
        audio = audio[0][0].data.cpu().float().numpy()

        timestamp = time.strftime("%m-%d_%H-%M", time.localtime())
        output_path = os.path.join(args.outdir, f"generated_{timestamp}.wav")
        write(
            output_path,
            hps.data.sampling_rate,
            audio,
        )
