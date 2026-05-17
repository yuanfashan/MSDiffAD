import sys

sys.path.append('./')
import time
import functools
import torch
import numpy as np
from tqdm import tqdm
import numpy as np
import torch
from tqdm import tqdm, trange
from model.diffusion.diffsion_augment_model import DiffudionAugment
from model.diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from utils.diff_utils import get_full_sort_score, EarlyStopping
from model.diffusion import gaussian_diffusion as gd
from utils.step_sample import UniformSampler
import torch.nn as nn
from info_nce import InfoNCE



class Trainer:
    def __init__(self, args, device, generator):

        self.args = args
        self.device = device
        self.start_epoch = 0  # define the start epoch for keepon trainingzhonss

        self.loss_func = torch.nn.BCEWithLogitsLoss()
        self.generator = generator
        self.train_dataloader = generator.train_dataloader
        self.valid_dataloader = generator.valid_dataloader
        self.test_dataloader = generator.test_dataloader
        self.item_size = 100002
        self.args.item_size = 100002
        self.generator = generator
        self.criterion = nn.CrossEntropyLoss(ignore_index=0)
        self.contrastive_loss = InfoNCE(0.05)

        self._create_model()
        self._set_optimizer()
        # self._set_stopper()

    def _create_model(self):
        self.model = DiffudionAugment(self.device, self.args)
        self.model.to(self.device)

        betas = gd.get_named_beta_schedule(self.args.noise_schedule, self.args.diffusion_steps)
        timestep_respacing = [self.args.diffusion_steps]
        self.diffusion = SpacedDiffusion(
            use_timesteps=space_timesteps(self.args.diffusion_steps, timestep_respacing),
            betas=betas,
            rescale_timesteps=self.args.rescale_timesteps,
            predict_xstart=self.args.predict_xstart,
            learn_sigmas=self.args.learn_sigma,
            sigma_small=self.args.sigma_small,
            use_kl=self.args.use_kl,
            rescale_learned_sigmas=self.args.rescale_learned_sigmas
        )
        self.schedule_sampler = UniformSampler(self.diffusion)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.model.parameters(),
                                          lr=self.args.learning_rate,
                                          # betas=(0.9, 0.999),
                                          weight_decay=self.args.weight_decay)

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):

        tr_loss = 0
        tr_diff_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        train_time = []

        self.model.train()
        prog_iter = tqdm(self.train_dataloader, leave=False, desc='Training')

        for batch in prog_iter:
            train_start = time.time()

            #  input_idss输入序列、attention_mask掩码（padding位置为0，其余位置为1）、masked_indices0选择扩散的位置
            input_ids, attention_mask, masked_indices0, intensity = \
                batch["input_ids"].to(self.device), \
                    batch["attention_mask"].to(self.device), \
                    batch["masked_indices0"].to(self.device), \
                    batch["intensity"].to(self.device)

            self.optimizer.zero_grad()

            t, weights = self.schedule_sampler.sample(input_ids.shape[0], self.device)
            compute_losses = functools.partial(
                self.diffusion.training_losses,
                self.model,
                t,
                input_ids,
                masked_indices0.long(),
                attention_mask
            )
            #  此处为基于上下文感知扩散
            diff_mse_loss, diff_nll_loss, aug_seq1, aug_seq2 = compute_losses()
            loss = diff_mse_loss + diff_nll_loss

            loss.backward()
            self.optimizer.step()

            tr_diff_loss += diff_nll_loss / len(self.train_dataloader)
            # tr_cl_loss += contrastive_loss / len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end - train_start)
        print(f' epoch {epoch}: diff_loss {tr_diff_loss:.4f}', end='   ')
        print(f'cl_loss {tr_cl_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}')

        # **保存 SpacedDiffusion 模型的状态
        if epoch % 10 == 0:  # 每10个epoch保存一次
            print(f"Saving SpacedDiffusion model at epoch {epoch}...")
            diffusion_state_dict = self.diffusion.state_dict()
            diff_augment_state_dict = self.model.state_dict()
            torch.save(diffusion_state_dict, f"./diff_path/spaced_diffusion_epoch_{epoch}.pt")
            torch.save(diff_augment_state_dict, f"./diff_path/diff_augment_epoch_{epoch}.pt")

    def train(self):
        print("********** Running training **********")
        train_time = []
        early_stopping = EarlyStopping(self.args.checkpoint_path, patience=40, verbose=True)

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)

            train_time.append(t)

            # if epoch % 10 == 0:
            #     self.eval(epoch, test=False)  #valid
            #
            #     self.eval(epoch, test=True)  #test
