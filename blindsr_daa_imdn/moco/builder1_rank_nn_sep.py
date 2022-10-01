# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved
import torch
import torch.nn as nn

class MoCo(nn.Module):
    """
    Build a MoCo model with: a query encoder, a key encoder, and a queue
    https://arxiv.org/abs/1911.05722
    """
    def __init__(self, base_encoder, base_ranker, dim=256, K=16*256, m=0.999, T=0.07, mlp=False):
        """
        dim: feature dimension (default: 128)
        K: queue size; number of negative keys (default: 65536)
        m: moco momentum of updating key encoder (default: 0.999)
        T: softmax temperature (default: 0.07)
        """
        super(MoCo, self).__init__()

        self.K = K
        self.m = m
        self.T = T

        self.encoder_q = base_encoder()
        self.ranker_q = base_ranker()

        self.max0 = nn.ReLU()
        self.rankloss = nn.MarginRankingLoss(margin=0.5)
        self.l1_loss = nn.L1Loss()


    def forward(self, im,
                lam1_qk=None, lam2_qk=None, noise_qk=None,
                lam1_k1=None, lam2_k2=None, noise_k3=None,
                is_bic=True):
        """
        Input:
            im_q: a batch of query images
            im_k: a batch of key images
        Output:
            logits, targets
        """
        if self.training:

            if is_bic == False:

                # compute query features
                im_q = im[:, 0, ...]                          # b, c, h, w
                im_k = im[:, 1, ...]                           # equal

                im_k1 = im[:, 2, ...]                            # b, c, h, w  lam1
                im_k2 = im[:, 3, ...]                            # b, c, h, w  lam2
                im_k3 = im[:, 4, ...]                            # b, c, h, w  noise
                im_kb = im[:, 5, ...]                            # b, c, h, w  bic

                embedding = self.encoder_q(im_q)  # queries: NxC, s_q: N
                score_q = self.ranker_q(embedding)

                # rank
                # compute rank loss
                score_ku = self.ranker_q(self.encoder_q(im_k))  # keys: NxC, s_k: N

                score_k1u = self.ranker_q(self.encoder_q(im_k1))  # keys1: NxC, s_k1: N

                score_k2u = self.ranker_q(self.encoder_q(im_k2))  # keys1: NxC, s_k1: N

                score_k3u = self.ranker_q(self.encoder_q(im_k3))  # keys1: NxC, s_k1: N

                score_kbu = self.ranker_q(self.encoder_q(im_kb))  # keys1: NxC, s_k1: N


                B = im_q.size()[0]
                loss_rank_all = 0.0

                QN = 3 * B

                # lam1 k1
                lam1_dd = lam1_qk - lam1_k1

                id_p = (lam1_dd > 0)
                if id_p.sum() > 0:
                    # y = torch.ones(id_p.sum(), requires_grad=False).cuda().float()
                    y = torch.ones(id_p.sum()).cuda().float()
                    rank_p1_l1 = self.rankloss(score_q[id_p], score_k1u[id_p], y)

                    loss_rank_all = loss_rank_all + rank_p1_l1

                id_n = (lam1_dd < 0)
                if id_n.sum() > 0:
                    # y = torch.ones(id_n.sum(), requires_grad=False).cuda().float() * -1.0
                    y = torch.ones(id_n.sum()).cuda().float() * -1.0
                    rank_n1_l1 = self.rankloss(score_q[id_n], score_k1u[id_n], y)

                    loss_rank_all = loss_rank_all + rank_n1_l1

                # lam2 k2
                lam2_dd = lam2_qk - lam2_k2

                id_p = (lam2_dd > 0)
                if id_p.sum() > 0:
                    # y = torch.ones(id_p.sum(), requires_grad=False).cuda().float()
                    y = torch.ones(id_p.sum()).cuda().float()
                    rank_p1_l2 = self.rankloss(score_q[id_p], score_k2u[id_p], y)

                    loss_rank_all = loss_rank_all + rank_p1_l2

                id_n = (lam2_dd < 0)
                if id_n.sum() > 0:
                    # y = torch.ones(id_n.sum(), requires_grad=False).cuda().float() * -1.0
                    y = torch.ones(id_n.sum()).cuda().float() * -1.0
                    rank_n1_l2 = self.rankloss(score_q[id_n], score_k2u[id_n], y)

                    loss_rank_all = loss_rank_all + rank_n1_l2

                # noise k3
                noise_dd = noise_qk - noise_k3

                id_p = (noise_dd > 0)
                if id_p.sum() > 0:
                    # y = torch.ones(id_p.sum(), requires_grad=False).cuda().float()
                    y = torch.ones(id_p.sum()).cuda().float()
                    rank_p1_n = self.rankloss(score_q[id_p], score_k3u[id_p], y)

                    loss_rank_all = loss_rank_all + rank_p1_n

                id_n = (noise_dd < 0)
                if id_n.sum() > 0:
                    # y = torch.ones(id_n.sum(), requires_grad=False).cuda().float() * -1.0
                    y = torch.ones(id_n.sum()).cuda().float() * -1.0
                    rank_n1_n = self.rankloss(score_q[id_n], score_k3u[id_n], y)

                    loss_rank_all = loss_rank_all + rank_n1_n

                # bic kb
                # y = torch.ones(B, requires_grad=False).cuda().float() * 1.0
                y = torch.ones(B).cuda().float() * 1.0
                rank_n1_b1 = self.rankloss(score_q, score_kbu, y)
                rank_n1_b2 = self.rankloss(score_ku, score_kbu, y)
                rank_n1_b3 = self.rankloss(score_k1u, score_kbu, y)
                rank_n1_b4 = self.rankloss(score_k2u, score_kbu, y)
                rank_n1_b5 = self.rankloss(score_k3u, score_kbu, y)

                rank_n1_b = (rank_n1_b1 + rank_n1_b2 + rank_n1_b3 + rank_n1_b4 + rank_n1_b5) * 0.2

                loss_rank_all = loss_rank_all + rank_n1_b

                # 3. score q = k
                eq_loss1 = self.l1_loss(score_q, score_ku.detach())
                eq_loss2 = self.l1_loss(score_ku, score_q.detach())

                loss_rank_all = loss_rank_all + (eq_loss1 + eq_loss2) * 5

                # 4. score_bic=0
                gt = torch.zeros_like(score_q)
                bic_loss1 = self.l1_loss(score_kbu, gt)

                loss_rank_all = loss_rank_all + bic_loss1 * 2


            # # bicubic
            # else:
            #     im_q = im[:, 0, ...]                          # b, c, h, w
            #     im_k = im[:, 1, ...]                           # equal

            #     B = im_q.size()[0]

            #     embedding, score_q = self.encoder_q(im_q)  # queries: NxC, s_q: N

            #     _, score_ku = self.encoder_q(im_k)  # keys: NxC, s_k: N

            #     # rank loss

            #     loss_rank_all = 0.0
            #     # 1. bicubic rank_loss = 0.0
            #     gt = torch.zeros_like(score_q)
            #     eq_loss1 = self.l1_loss(score_q, gt)
            #     eq_loss2 = self.l1_loss(score_ku, gt)

            #     loss_rank_all = loss_rank_all + (eq_loss1 + eq_loss2) * 2

            #     # 2. rank loss all neg
            #     # QN = 3 * B

            #     # score_all_q = torch.cat([score_q, score_q, score_q], dim=0)
            #     # score_all_ka = torch.cat([score_k1u, score_k2u, score_k3u], dim=0)

            #     # y = torch.ones(QN, requires_grad=False).cuda().float() * -1.0
            #     # rank_n1 = self.rankloss(score_all_q, score_all_ka, y)

            #     # loss_rank_all = loss_rank_all + rank_n1 * 0.2
                

            return embedding, loss_rank_all, score_q

        else:

            embedding = self.encoder_q(im)
            score = self.ranker_q(embedding)

            return embedding, score


# utils
@torch.no_grad()
def concat_all_gather(tensor):
    """
    Performs all_gather operation on the provided tensors.
    *** Warning ***: torch.distributed.all_gather has no gradient.
    """
    tensors_gather = [torch.ones_like(tensor)
        for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(tensors_gather, tensor, async_op=False)

    output = torch.cat(tensors_gather, dim=0)
    return output
