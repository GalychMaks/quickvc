import logging
import os
from pathlib import Path

import torch
from omegaconf import OmegaConf
from torch.nn import functional as F
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter

from quickvc.modules.models import (
    MultiPeriodDiscriminator,
    SynthesizerTrn,
)
from quickvc.modules.pqmf import PQMF
from quickvc.training.data_utils_new_new import (
    TextAudioSpeakerCollate,
    TextAudioSpeakerLoader,
)
from quickvc.training.losses import (
    discriminator_loss,
    feature_loss,
    generator_loss,
    kl_loss,
    subband_stft_loss,
)
from quickvc.utils.commons import clip_grad_value_, slice_segments
from quickvc.utils.mel_processing import mel_spectrogram_torch, spec_to_mel_torch
from quickvc.utils.utils import (
    latest_checkpoint_path,
    load_checkpoint,
    save_checkpoint,
    summarize,
)

torch.autograd.set_detect_anomaly(True)
torch.backends.cudnn.benchmark = True


logger = logging.getLogger(__name__)


class Trainer:
    def __init__(self, config_path: Path | str):
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.config = OmegaConf.load(config_path)
        self.config.model_dir = "logs/quickvc"
        self.global_step = 0
        logger.info(self.config)

        self.writer = SummaryWriter(log_dir=self.config.model_dir)
        self.writer_eval = SummaryWriter(
            log_dir=os.path.join(self.config.model_dir, "eval")
        )

        torch.manual_seed(self.config.train.seed)

        # Dataloaders
        collate_fn = TextAudioSpeakerCollate(self.config)
        train_dataset = TextAudioSpeakerLoader(
            self.config.data.training_files, self.config
        )
        self.train_loader = DataLoader(
            train_dataset,
            num_workers=8,
            shuffle=False,
            pin_memory=True,
            collate_fn=collate_fn,
        )

        eval_dataset = TextAudioSpeakerLoader(
            self.config.data.validation_files, self.config
        )
        self.eval_loader = DataLoader(
            eval_dataset,
            num_workers=8,
            shuffle=True,
            batch_size=1,
            pin_memory=False,
            drop_last=False,
        )

        # Networks
        self.net_g = SynthesizerTrn(
            self.config.data.filter_length // 2 + 1,
            self.config.train.segment_size // self.config.data.hop_length,
            **self.config.model,
        ).to(self.device)

        self.net_d = MultiPeriodDiscriminator(self.config.model.use_spectral_norm).to(
            self.device
        )

        # Optimizers
        self.optim_g = torch.optim.AdamW(
            self.net_g.parameters(),
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps,
        )
        self.optim_d = torch.optim.AdamW(
            self.net_d.parameters(),
            self.config.train.learning_rate,
            betas=self.config.train.betas,
            eps=self.config.train.eps,
        )

        # Load checkpoints
        try:
            _, _, _, self.epoch_str = load_checkpoint(
                latest_checkpoint_path(self.config.model_dir, "G_*.pth"),
                self.net_g,
                self.optim_g,
            )
            _, _, _, self.epoch_str = load_checkpoint(
                latest_checkpoint_path(self.config.model_dir, "D_*.pth"),
                self.net_d,
                self.optim_d,
            )
            self.global_step = (self.epoch_str - 1) * len(self.train_loader)
        except Exception:
            print("Training from scratch")
            self.epoch_str = 1
            self.global_step = 0

        # Schedulers
        self.scheduler_g = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_g,
            gamma=self.config.train.lr_decay,
            last_epoch=self.epoch_str - 2,
        )
        self.scheduler_d = torch.optim.lr_scheduler.ExponentialLR(
            self.optim_d,
            gamma=self.config.train.lr_decay,
            last_epoch=self.epoch_str - 2,
        )

    def train(self):
        # Training loop
        for epoch in range(self.epoch_str, self.config.train.epochs + 1):
            self.train_and_evaluate(epoch)

            self.scheduler_g.step()
            self.scheduler_d.step()

    def train_and_evaluate(
        self,
        epoch,
    ):
        tmp = 0
        tmp1 = 1000000000
        # train_loader.batch_sampler.set_epoch(epoch)

        self.net_g.train()
        self.net_d.train()
        for batch_idx, (c, spec, y) in enumerate(self.train_loader):
            spec, y = spec.to(self.device), y.to(self.device)

            c = c.to(self.device)
            mel = spec_to_mel_torch(
                spec,
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax,
            )

            (
                y_hat,
                y_hat_mb,
                ids_slice,
                z_mask,
                (z, z_p, m_p, logs_p, m_q, logs_q),
            ) = self.net_g(c, spec, g=None, mel=mel)

            mel = spec_to_mel_torch(
                spec,
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax,
            )
            y_mel = slice_segments(
                mel,
                ids_slice,
                self.config.train.segment_size // self.config.data.hop_length,
            )
            y_hat_mel = mel_spectrogram_torch(
                y_hat.squeeze(1),
                self.config.data.filter_length,
                self.config.data.n_mel_channels,
                self.config.data.sampling_rate,
                self.config.data.hop_length,
                self.config.data.win_length,
                self.config.data.mel_fmin,
                self.config.data.mel_fmax,
            )
            tmp = max(tmp, y.size()[2])
            tmp1 = min(tmp1, y.size()[2])
            y = slice_segments(
                y,
                ids_slice * self.config.data.hop_length,
                self.config.train.segment_size,
            )  # slice

            y_d_hat_r, y_d_hat_g, _, _ = self.net_d(y, y_hat.detach())
            loss_disc, losses_disc_r, losses_disc_g = discriminator_loss(
                y_d_hat_r, y_d_hat_g
            )
            loss_disc_all = loss_disc

            self.optim_d.zero_grad()
            loss_disc_all.backward()
            grad_norm_d = clip_grad_value_(self.net_d.parameters(), None)
            self.optim_d.step()

            # Generator
            y_d_hat_r, y_d_hat_g, fmap_r, fmap_g = self.net_d(y, y_hat)

            # loss_dur = torch.sum(l_length.float())
            loss_mel = F.l1_loss(y_mel, y_hat_mel) * self.config.train.c_mel
            loss_kl = kl_loss(z_p, logs_q, m_p, logs_p, z_mask) * self.config.train.c_kl

            loss_fm = feature_loss(fmap_r, fmap_g)
            loss_gen, losses_gen = generator_loss(y_d_hat_g)

            if self.config.model.mb_istft_vits:
                pqmf = PQMF(y.device)
                y_mb = pqmf.analysis(y)
                loss_subband = subband_stft_loss(self.config, y_mb, y_hat_mb)
            else:
                loss_subband = torch.tensor(0.0)

            loss_gen_all = (
                loss_gen + loss_fm + loss_mel + loss_kl + loss_subband
            )  # + loss_dur

            self.optim_g.zero_grad()
            loss_gen_all.backward()
            grad_norm_g = clip_grad_value_(self.net_g.parameters(), None)
            self.optim_g.step()

            if self.global_step % self.config.train.log_interval == 0:
                lr = self.optim_g.param_groups[0]["lr"]
                losses = [
                    loss_disc,
                    loss_gen,
                    loss_fm,
                    loss_mel,
                    loss_kl,
                    loss_subband,
                ]
                logger.info(
                    f"Train Epoch: {epoch} [{100.0 * batch_idx / len(self.train_loader):.0f}%]"
                )
                logger.info([x.item() for x in losses] + [self.global_step, lr])

                scalar_dict = {
                    "loss/g/total": loss_gen_all,
                    "loss/d/total": loss_disc_all,
                    "learning_rate": lr,
                    "grad_norm_d": grad_norm_d,
                    "grad_norm_g": grad_norm_g,
                }
                scalar_dict.update(
                    {
                        "loss/g/fm": loss_fm,
                        "loss/g/mel": loss_mel,
                        "loss/g/kl": loss_kl,
                        "loss/g/subband": loss_subband,
                    }
                )

                scalar_dict.update({f"loss/g/{i}": v for i, v in enumerate(losses_gen)})
                scalar_dict.update(
                    {f"loss/d_r/{i}": v for i, v in enumerate(losses_disc_r)}
                )
                scalar_dict.update(
                    {f"loss/d_g/{i}": v for i, v in enumerate(losses_disc_g)}
                )

                summarize(
                    writer=self.writer,
                    global_step=self.global_step,
                    scalars=scalar_dict,
                )

            if self.global_step % self.config.train.eval_interval == 0:
                self.evaluate()

                save_checkpoint(
                    self.net_g,
                    self.optim_g,
                    self.config.train.learning_rate,
                    epoch,
                    os.path.join(self.config.model_dir, f"G_{self.global_step}.pth"),
                )
                save_checkpoint(
                    self.net_d,
                    self.optim_d,
                    self.config.train.learning_rate,
                    epoch,
                    os.path.join(self.config.model_dir, f"D_{self.global_step}.pth"),
                )

            self.global_step += 1

        logger.info(f"====> Epoch: {epoch}")
        print(tmp, tmp1)

    @torch.no_grad()
    def evaluate(self):
        self.net_g.eval()

        # Predefine variables to avoid "possibly unbound" warnings
        c = spec = y = None

        for batch_idx, (c_batch, spec_batch, y_batch) in enumerate(self.eval_loader):
            spec = spec_batch[:1].to(self.device)
            y = y_batch[:1].to(self.device)
            c = c_batch[:1].to(self.device)
            break

        if spec is None or c is None or y is None:
            logger.error("Couldn't load batch from eval_loader")
            return

        mel = spec_to_mel_torch(
            spec,
            self.config.data.filter_length,
            self.config.data.n_mel_channels,
            self.config.data.sampling_rate,
            self.config.data.mel_fmin,
            self.config.data.mel_fmax,
        )
        y_hat = self.net_g.infer(c, mel=mel)

        audio_dict = {"gen/audio": y_hat[0], "gt/audio": y[0]}

        summarize(
            writer=self.writer_eval,
            global_step=self.global_step,
            audios=audio_dict,
            audio_sampling_rate=self.config.data.sampling_rate,
        )
