import torch
from torch import nn
import model.common as common
import torch.nn.functional as F
from moco.builder1_rank_nn_sep import MoCo

import pickle
import torch.nn.init as init

from model.IMBD_mod_tt import IMDN
import collections


def make_model(args):
    return BlindSR(args)


# class IMBD(nn.Module):
#     def __init__(self, config):
#         super(IMBD, self).__init__()

#         self.model = IMDN()

#         self.criterion = nn.L1Loss(reduction='mean')

#         all_params = self.named_parameters()

#         training_params = []

#         # for name, params in all_params:
#         #     if 'adapt' not in name:
#         #         params.requires_grad = False
#         #     else:
#         #         training_params.append(name)

#         # print(training_params)

#     def forward(self, x, dr, gt=None):
        
#         out, diff = self.model(x, dr)

#         return out, diff


class Encoder(nn.Module):
    def __init__(self):
        super(Encoder, self).__init__()

        self.E = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(64, 128, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(128, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.Conv2d(256, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.1, True),
            nn.AdaptiveAvgPool2d(1),
        )

        # self.mlp1 = nn.Sequential(
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 100),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(100, 1),
        #     # nn.ReLU(True)
        # )

    def forward(self, x):
        fea = self.E(x).squeeze(-1).squeeze(-1)
        # score = self.mlp1(fea).squeeze(-1)

        # return fea, score
        return fea


class Ranker(nn.Module):
    def __init__(self):
        super(Ranker, self).__init__()

        self.R = nn.Sequential(
            nn.LeakyReLU(0.1, True),
            nn.Linear(256, 128),
            nn.LeakyReLU(0.1, True),
            nn.Linear(128, 64),
            nn.LeakyReLU(0.1, True),
            nn.Linear(64, 32),
            nn.LeakyReLU(0.1, True),
            nn.Linear(32, 1),
            # nn.ReLU(True)
        )

        # self.mlp1 = nn.Sequential(
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(256, 100),
        #     nn.LeakyReLU(0.1, True),
        #     nn.Linear(100, 1),
        #     # nn.ReLU(True)
        # )

    def forward(self, x):
        # fea = self.E(x).squeeze(-1).squeeze(-1)
        score = self.R(x).squeeze(-1)

        # return fea, score
        return score


class BlindSR(nn.Module):
    def __init__(self, args, pretrain=True):
        super(BlindSR, self).__init__()

        # Generator
        self.G = IMDN()

        # Encoder
        self.E = MoCo(base_encoder=Encoder, base_ranker=Ranker)

        if pretrain:
            modelg_path = args.model_pre_train
            loaded_model = torch.load(modelg_path)
            loaded_model_nn = collections.OrderedDict()
            for key, value in loaded_model.items():
                name = key.split('.')[1:]
                tag = "."
                new_key = tag.join(name)
                loaded_model_nn[new_key] = value
            self.G.load_state_dict(loaded_model_nn, strict=False)

    # def adapt_params(self):
    #
    #     G_params = self.G.named_parameters()
    #     for name, params in G_params:
    #         if 'adapt' in name:
    #             # print(name)
    #             yield params
    #
    # def noadapt_params(self):
    #
    #     G_params = self.G.named_parameters()
    #     for name, params in G_params:
    #         if 'adapt' not in name:
    #             # print(name)
    #             yield params

    # def optim_parameters(self, args):
    #     G_params = filter(lambda x: x.requires_grad, self.G.parameters())
    #     E_params = filter(lambda x: x.requires_grad, self.E.encoder_q.parameters())
    #     return [{'params': G_params, 'lr': args.lr_sr},
    #             {'params': E_params, 'lr': args.lr_encoder}, ]


    # def optim_parameters_R(self, args):
    #     R_params = filter(lambda x: x.requires_grad, self.E.ranker_q.parameters())
    #     return [{'params': E_params, 'lr': args.lr_sr},
    #             {'params': E_params, 'lr': args.lr_encoder}, ]

    # def optim_parameters_s2(self, args):
    #     G_params = filter(lambda x: x.requires_grad, self.G.parameters())
    #     E_params = filter(lambda x: x.requires_grad, self.E.parameters())
    #     return [{'params': G_params, 'lr': args.lr_sr},
    #             {'params': E_params, 'lr': args.lr_encoder}, ]


    def forward(self, x, 
                lam1_qk=None, lam2_qk=None, noise_qk=None, 
                lam1_k1=None, lam2_k2=None, noise_k3=None,
                is_bicubic=False):
        if self.training:

            # degradation-aware represenetion learning
            fea, loss_rank, score = self.E(x, 
                                                        lam1_qk, lam2_qk, noise_qk,
                                                        lam1_k1, lam2_k2, noise_k3,
                                                        is_bicubic)

            # degradation-aware SR
            # x2 = torch.cat([x_query, x_key], dim=0)
            # fea2 = torch.cat([fea, fea], dim=0)
            x_query = x[:, 0, ...]                          # b, c, h, w
            # x_key = x[:, 1, ...]
            # x_keya = x[:, 2:, ...]                            # b, c, h, w
            sr, diff = self.G(x_query / 255.0, fea)

            return sr, diff, loss_rank, score

        else:

            fea, score = self.E(x, is_bicubic)
            sr, diff = self.G(x / 255.0, fea)

            return sr, diff, score
