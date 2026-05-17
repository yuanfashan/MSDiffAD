import sys

sys.path.append('./')
import time
import functools
import torch
from tqdm import tqdm, trange

from model.diffusion.diffsion_augment_model import DiffudionAugment
from model.diffusion.gaussian_diffusion import SpacedDiffusion, space_timesteps
from utils.diff_utils import get_full_sort_score, EarlyStopping
from model.diffusion import gaussian_diffusion as gd
from utils.step_sample import UniformSampler


import torch.nn as nn
from info_nce import InfoNCE
from utils.ms_utils import ParseOrbitrap, ModelEmbed, SearchTop_MemoryEfficient
from model.representation.representation_model import MSDiffusion



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
        model_file = 'D:/Projects/MSDiffAD/train/diff_path/DiffudionAugment.pt'
        self.model.load_state_dict(torch.load(model_file))

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
        model_file = 'D:/Projects/MSDiffAD/train/diff_path/spaced_diffusion.pt'
        self.diffusion.load_state_dict(torch.load(model_file))
        self.schedule_sampler = UniformSampler(self.diffusion)
        self.msencoder = MSDiffusion(100002, 512, 6, 16, 0, 100, 3)
        self.msencoder.to(self.device)

    def _set_optimizer(self):

        self.optimizer = torch.optim.Adam(self.msencoder.parameters(),
                                          lr=1e-4,
                                          # betas=(0.9, 0.999),
                                          weight_decay=self.args.weight_decay)

    def _train_one_epoch(self, epoch, only_bert_train=False, is_diffusion_train=True):
        tr_loss = 0
        tr_diff_loss = 0
        tr_mlm_loss = 0
        tr_cl_loss = 0
        tr_mcl_loss = 0
        train_time = []

        self.model.eval()
        self.diffusion.eval()
        self.msencoder.train()
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
            _, _, aug_seq1, aug_seq2 = compute_losses()

            # 前向传播
            results_a = self.msencoder(aug_seq1, aug_seq2, intensity)
            logits_lm1,logits_lm2, mask_token, pool1, pool2, logits_lm3, mask_token3, pool3 = results_a
            # 计算重建损失
            reconstruction_loss1_a = self.criterion(logits_lm1.view(-1, logits_lm1.size(-1)),
                                                    mask_token.view(-1))
            reconstruction_loss2_a = self.criterion(logits_lm2.view(-1, logits_lm2.size(-1)),
                                                    mask_token.view(-1))
            reconstruction_loss3_a = self.criterion(logits_lm3.view(-1, logits_lm3.size(-1)),
                                                    mask_token3.view(-1))
            mlm_loss_a = (reconstruction_loss1_a + reconstruction_loss2_a + reconstruction_loss3_a) / 3
            # 计算对比损失
            contrastive_loss_a = self.contrastive_loss(pool1.squeeze(), pool2.squeeze())
            m_contrastive_loss = self.contrastive_loss(pool1.squeeze(), pool3.squeeze())

            # results_b = self.msencoder(aug_seq1, intensity)
            # logits_lm1_b, mask_token1_b, pool1_b, logits_lm2_b, mask_token2_b, pool2_b = results_b
            # # 计算重建损失
            # reconstruction_loss1_b = self.criterion(logits_lm1_b.view(-1, logits_lm1_b.size(-1)),
            #                                         mask_token1_b.view(-1))
            # reconstruction_loss2_b = self.criterion(logits_lm2_b.view(-1, logits_lm2_b.size(-1)),
            #                                         mask_token2_b.view(-1))
            # mlm_loss_b = (reconstruction_loss1_b + reconstruction_loss2_b) / 2
            # # 计算对比损失
            # contrastive_loss_b = self.contrastive_loss(pool1_b.squeeze(), pool2_b.squeeze())
            # contrastive_loss_c = self.contrastive_loss(pool1_a.squeeze(), pool1_b.squeeze())
            mlm_loss = mlm_loss_a
            contrastive_loss = contrastive_loss_a
            # 总损失
            loss = mlm_loss + 20*contrastive_loss + m_contrastive_loss
            loss.backward()
            self.optimizer.step()

            tr_mlm_loss += mlm_loss / len(self.train_dataloader)
            tr_cl_loss += contrastive_loss / len(self.train_dataloader)
            tr_mcl_loss += m_contrastive_loss/ len(self.train_dataloader)
            tr_loss += loss.item() / len(self.train_dataloader)

            train_end = time.time()
            train_time.append(train_end - train_start)
        # if epoch %10==0:
        #     print("aug_seq1", aug_seq1[masked_indices0])
        #     print("aug_seq2", aug_seq2[masked_indices0])
        print(f' epoch {epoch}: mlm_loss {tr_mlm_loss:.4f}', end='   ')
        print(f'cl_loss {tr_cl_loss:.4f}', end='   ')
        print(f'cl_loss {tr_mcl_loss:.4f}', end='   ')
        print(f'total_loss {tr_loss:.4f}')


    def train(self):
        print("********** Running training **********")
        train_time = []
        early_stopping = EarlyStopping(self.args.checkpoint_path, patience=40, verbose=True)

        for epoch in trange(self.start_epoch, self.start_epoch + int(self.args.epochs), desc="Epoch"):
            t = self._train_one_epoch(epoch, only_bert_train=False, is_diffusion_train=True)

            train_time.append(t)


    def test_step(self, best_accuracy, test_ref, test_query, smiles1, smiles2, batch_size=128):
        self.msencoder.eval()
        with torch.no_grad():
            test_ref_arr = ModelEmbed(self.msencoder, test_ref, batch_size)
            test_query_arr = ModelEmbed(self.msencoder, test_query, batch_size)
            top = SearchTop_MemoryEfficient(test_ref_arr, test_query_arr, smiles1, smiles2)
            top1 = top[0]
            pass
            # **保存 msencoder 模型的状态**
            if top1 > best_accuracy:
                best_accuracy = top1
                print(f"Saving msencoder model at epoch..")
                msencoder_state_dict = self.msencoder.state_dict()
                torch.save(msencoder_state_dict, f"./ms_repre_pth/msdiffad_encoder.pt")
            print(f"Test Top-1 Accuracy: {top1 * 100:.2f}%")
        return top1, best_accuracy

