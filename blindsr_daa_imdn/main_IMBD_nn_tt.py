from option_IMBD_nn_tt import args
import torch
import utility1
import data_nn as data
import model
import loss
from trainer_nn_imbd_tt1_test1 import Trainer

import os
import numpy as np
import random


def init_random_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)

    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


if __name__ == '__main__':
    # torch.manual_seed(args.seed)

    os.environ['CUDA_VISIBLE_DEVICES'] = "2"

    # torch.manual_seed(args.seed)
    rank = 0
    init_random_seed(args.seed + rank)

    checkpoint = utility1.checkpoint(args)
    if checkpoint.ok:
        loader = data.Data(args)
        model = model.Model(args, checkpoint)
        loss = loss.Loss(args, checkpoint) if not args.test_only else None
        t = Trainer(args, loader, model, loss, checkpoint)

        ii = 1
        while not t.terminate():
            t.train()

            if (ii % 10 == 0) & (ii > args.epochs_encoder):
                t.testn()
                t.testn_tt()
                t.test1()

            ii = ii+1

        checkpoint.done()
