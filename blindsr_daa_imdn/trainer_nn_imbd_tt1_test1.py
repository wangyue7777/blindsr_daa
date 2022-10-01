import os
import utility
import torch
from decimal import Decimal
import torch.nn.functional as F
from utils import util

import math


class Trainer():
    def __init__(self, args, loader, my_model, my_loss, ckp):
        self.args = args
        self.scale = args.scale

        self.ckp = ckp
        self.loader_train = loader.loader_train
        self.loader_test = loader.loader_test
        self.loader_test_tt = loader.loader_test_tt
        self.loader_test1 = loader.loader_test1
        self.model = my_model
        self.model_E = torch.nn.DataParallel(self.model.get_model().E, range(self.args.n_GPUs))
        self.loss = my_loss
        self.contrast_loss = torch.nn.CrossEntropyLoss().cuda()
        self.optimizer_EG = utility.make_optimizer_EG(args, self.model)
        self.optimizer_R = utility.make_optimizer_R(args, self.model)
        self.scheduler1, self.scheduler2 = utility.make_scheduler2(args, self.optimizer_EG, self.optimizer_R)

        if self.args.load != '.':
            self.optimizer_EG.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_EG.pt'))
            )
            self.optimizer_R.load_state_dict(
                torch.load(os.path.join(ckp.dir, 'optimizer_r.pt'))
            )
            for _ in range(len(ckp.log)): 
                self.scheduler1.step()
                self.scheduler2.step()

    def train(self):
        self.scheduler1.step()
        self.scheduler2.step()
        self.loss.step()
        epoch = self.scheduler1.last_epoch + 1

        # lr stepwise
        if epoch <= self.args.epochs_encoder:
            lr = self.args.lr_encoder * (self.args.gamma_encoder ** (epoch // self.args.lr_decay_encoder))
            for param_group in self.optimizer_EG.param_groups:
                param_group['lr'] = lr
            for param_group in self.optimizer_R.param_groups:
                param_group['lr'] = lr
        else:
            R_params = self.model.get_model().E.ranker_q.parameters()
            for params in R_params:
                params.requires_grad = False

            lr = self.args.lr_sr * (self.args.gamma_sr ** ((epoch - self.args.epochs_encoder) // self.args.lr_decay_sr))
            for param_group in self.optimizer_EG.param_groups:
                param_group['lr'] = lr

        self.ckp.write_log('[Epoch {}]\tLearning rate: {:.2e}'.format(epoch, Decimal(lr)))
        self.loss.start_log()
        self.model.train()

        timer = utility.timer()
        losses_rank, losses_sr = utility.AverageMeter(), utility.AverageMeter()

        for batch, (hr, _, idx_scale) in enumerate(self.loader_train):
            hr = hr.cuda()                              # b, n, c, h, w

            B = hr.size()[0]

            timer.tic()

            lambda_1 = torch.rand(B).cuda() * (self.args.lambda_max - self.args.lambda_min) + self.args.lambda_min
            lambda_2 = torch.rand(B).cuda() * (self.args.lambda_max - self.args.lambda_min) + self.args.lambda_min
            noise = torch.rand(B).cuda() * self.args.noise
            theta = torch.rand(B).cuda()

            degrade_qk = util.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type=self.args.blur_type,
                sig=4.0,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                noise=noise,
                theta=theta
            )

            lr_q, b_kernels_q = degrade_qk(hr[:, 0, ...].unsqueeze(1), random=False)                 # bn, c, h, w
            lr_k, b_kernels_k = degrade_qk(hr[:, 1, ...].unsqueeze(1), random=False)                 # bn, c, h, w

            lambda_1n = torch.rand(B).cuda() * (self.args.lambda_max - self.args.lambda_min) + self.args.lambda_min
            lambda_2n = torch.rand(B).cuda() * (self.args.lambda_max - self.args.lambda_min) + self.args.lambda_min
            noise_n = torch.rand(B).cuda() * self.args.noise

            degrade_k1 = util.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type=self.args.blur_type,
                sig=4.0,
                lambda_1=lambda_1n,
                lambda_2=lambda_2,
                noise=noise,
                theta=theta
            )

            lr_k1, b_kernels_k1 = degrade_k1(hr[:, 1, ...].unsqueeze(1), random=False)                 # bn, c, h, w

            degrade_k2 = util.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type=self.args.blur_type,
                sig=4.0,
                lambda_1=lambda_1,
                lambda_2=lambda_2n,
                noise=noise,
                theta=theta
            )

            lr_k2, b_kernels_k2 = degrade_k2(hr[:, 1, ...].unsqueeze(1), random=False)                 # bn, c, h, w

            degrade_k3 = util.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type=self.args.blur_type,
                sig=4.0,
                lambda_1=lambda_1,
                lambda_2=lambda_2,
                noise=noise_n,
                theta=theta
            )

            lr_k3, b_kernels_k3 = degrade_k3(hr[:, 1, ...].unsqueeze(1), random=False)                 # bn, c, h, w

            degrade_bic = util.SRMDPreprocessing(
                self.scale[0],
                kernel_size=self.args.blur_kernel,
                blur_type='iso_gaussian',
                sig=0.0,
                noise=0.0
            )

            lr_bic, b_kernels = degrade_bic(hr[:, 1, ...].unsqueeze(1), random=False)                 # bn, c, h, w

            lr_rand = torch.cat((lr_q, lr_k, lr_k1, lr_k2, lr_k3, lr_bic), dim=1)

            # forward
            ## train degradation encoder
            if epoch <= self.args.epochs_encoder:

                self.optimizer_EG.zero_grad()
                self.optimizer_R.zero_grad()

                _, loss_rank_rand, score_rand = self.model_E(im=lr_rand, lam1_qk=lambda_1, lam2_qk=lambda_2, noise_qk=noise,
                                                            lam1_k1=lambda_1n, lam2_k2=lambda_2n, noise_k3=noise_n, is_bic=False)

                loss = loss_rank_rand

                losses_rank.update(loss_rank_rand.item())

                loss.backward()
                self.optimizer_EG.step()
                self.optimizer_R.step()

            ## train the whole network
            else:

                self.optimizer_EG.zero_grad()

                sr_rand, diff_rand, loss_rank_rand, score_rand = self.model(lr_rand, lam1_qk=lambda_1, lam2_qk=lambda_2, noise_qk=noise,
                                    lam1_k1=lambda_1n, lam2_k2=lambda_2n, noise_k3=noise_n, is_bicubic=False)

                loss_SR = self.loss(sr_rand, hr[:,0,...] / 255.0) * 255.0
                # loss_constrast = self.contrast_loss(output, target)
                loss = loss_rank_rand * 0.2 + loss_SR

                losses_sr.update(loss_SR.item())
                losses_rank.update(loss_rank_rand.item())

                loss.backward()
                self.optimizer_EG.step()

            # backward
            
            timer.hold()

            if epoch <= self.args.epochs_encoder:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:03d}][{:04d}/{:04d}]\t'
                        'Loss [contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_rank.avg,
                            timer.release()
                        ))
            else:
                if (batch + 1) % self.args.print_every == 0:
                    self.ckp.write_log(
                        'Epoch: [{:04d}][{:04d}/{:04d}]\t'
                        'Loss [SR loss:{:.3f} | contrastive loss: {:.3f}]\t'
                        'Time [{:.1f}s]'.format(
                            epoch, (batch + 1) * self.args.batch_size, len(self.loader_train.dataset),
                            losses_sr.avg, losses_rank.avg,
                            timer.release(),
                        ))

        self.loss.end_log(len(self.loader_train))

        # save model
        target = self.model.get_model()
        model_dict = target.state_dict()
        keys = list(model_dict.keys())
        # for key in keys:
        #     if 'E.encoder_k' in key or 'queue' in key:
        #         del model_dict[key]
        torch.save(
            model_dict,
            os.path.join(self.ckp.dir, 'model', 'model_{}.pt'.format(epoch))
        )


    def test(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                degrade = util.SRMDPreprocessing(
                    self.scale[0],
                    kernel_size=self.args.blur_kernel,
                    blur_type=self.args.blur_type,
                    sig=4.0,
                    lambda_1=4.0,
                    lambda_2=3.0,
                    theta=165.0/180.0,
                    noise=10.0
                )

                for idx_img, (hr, filename, _) in enumerate(self.loader_test):
                    hr = hr.cuda()                      # b, 1, c, h, w
                    hr = self.crop_border(hr, scale)
                    lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    hr = hr[:, 0, ...]                  # b, c, h, w

                    # inference
                    timer_test.tic()
                    sr, diff, score = self.model(lr[:, 0, ...])
                    timer_test.hold()

                    sr = utility.quantize(sr * 255.0, self.args.rgb_range)
                    hr = utility.quantize(hr, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        # self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_wgt(filename, sr, hr, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
        
        self.model.train()


    def testn(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                for idx_img, (hr_img, lr_img, filename, _) in enumerate(self.loader_test):
                    hr_img = hr_img.cuda()
                    lr_img = lr_img.cuda()  # b, 1, c, h, w
                    hr_img = self.crop_border(hr_img, scale)
                    # lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    hr_img = hr_img[:, 0, ...]                  # b, c, h, w
                    lr_img = lr_img[:, 0, ...]            # b, c, h, w

                    # inference
                    timer_test.tic()
                    sr, diff, score = self.model(lr_img)
                    timer_test.hold()

                    sr = utility.quantize(sr * 255.0, self.args.rgb_range)
                    hr = utility.quantize(hr_img, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test.dataset.benchmark
                    )

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        # self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_wgt(filename, sr, hr, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test),
                        eval_ssim / len(self.loader_test),
                    ))
        
        self.model.train()


    def testn_tt(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()

        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                for idx_img, (hr_img, lr_img, filename, _) in enumerate(self.loader_test_tt):
                    hr_img = hr_img.cuda()
                    lr_img = lr_img.cuda()  # b, 1, c, h, w
                    hr_img = self.crop_border(hr_img, scale)
                    # lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    hr_img = hr_img[:, 0, ...]                  # b, c, h, w
                    lr_img = lr_img[:, 0, ...]            # b, c, h, w

                    # inference
                    timer_test.tic()
                    sr, diff, score = self.model(lr_img)
                    timer_test.hold()

                    sr = utility.quantize(sr * 255.0, self.args.rgb_range)
                    hr = utility.quantize(hr_img, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr, hr, scale, self.args.rgb_range,
                        benchmark=self.loader_test_tt.dataset.benchmark
                    )
                    eval_ssim += utility.calc_ssim(
                        sr, hr, scale,
                        benchmark=self.loader_test_tt.dataset.benchmark
                    )

                    # save results
                    if self.args.save_results:
                        save_list = [sr]
                        filename = filename[0]
                        # self.ckp.save_results(filename, save_list, scale)
                        self.ckp.save_results_wgt(filename, sr, hr, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test_tt)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test,
                        scale,
                        eval_psnr / len(self.loader_test_tt),
                        eval_ssim / len(self.loader_test_tt),
                    ))
        
        self.model.train()


    def test1(self):
        self.ckp.write_log('\nEvaluation:')
        self.ckp.add_log(torch.zeros(1, len(self.scale)))
        self.model.eval()

        timer_test = utility.timer()
        
        with torch.no_grad():
            for idx_scale, scale in enumerate(self.scale):
                # self.loader_test.dataset.set_scale(idx_scale)
                eval_psnr = 0
                eval_ssim = 0

                # degrade = util.SRMDPreprocessing(
                #     self.scale[0],
                #     kernel_size=self.args.blur_kernel,
                #     blur_type=self.args.blur_type,
                #     sig=self.args.sig,
                #     noise=self.args.noise
                # )

                for idx_img, (hr_img, lr_img, filename, _) in enumerate(self.loader_test1):
                    hr_img = hr_img.cuda()
                    lr_img = lr_img.cuda()  # b, 1, c, h, w
                    hr_img = self.crop_border(hr_img, scale)
                    # lr, _ = degrade(hr, random=False)   # b, 1, c, h, w
                    hr_img = hr_img[:, 0, ...]                  # b, c, h, w
                    lr_img = lr_img[:, 0, ...]

                    sc = round(scale)

                    # lr_img_rep = lr_img * 255.0

                    _, C, H, W = lr_img.size()

                    timer_test.tic()

                    # if H * W > 100000:
                    #     rf = 20

                    #     s_v = math.ceil(H * 0.5)
                    #     s_h = math.ceil(W * 0.5)

                    #     # lq_TL = lr_img[:, :, 0: s_v + rf, 0: s_h + rf]
                    #     # lq_BL = lr_img[:, :, s_v - rf:, 0: s_h + rf]
                    #     # lq_TR = lr_img[:, :, 0: s_v + rf, s_h - rf:]
                    #     # lq_BR = lr_img[:, :, s_v - rf:, s_h - rf:]

                    #     lq_TL_rep = lr_img[:, :, 0: s_v + rf, 0: s_h + rf]
                    #     lq_BL_rep = lr_img[:, :, s_v - rf:, 0: s_h + rf]
                    #     lq_TR_rep = lr_img[:, :, 0: s_v + rf, s_h - rf:]
                    #     lq_BR_rep = lr_img[:, :, s_v - rf:, s_h - rf:]

                    #     out_TL, diff, score = self.model(lq_TL_rep, is_bicubic=False)
                    #     out_TL = out_TL[:, :, 0:s_v * sc, 0:s_h * sc]

                    #     out_BL, diff, score = self.model(lq_BL_rep, is_bicubic=False)
                    #     out_BL = out_BL[:, :, rf * sc:, 0:s_h * sc]

                    #     out_TR, diff, score = self.model(lq_TR_rep, is_bicubic=False)
                    #     out_TR = out_TR[:, :, 0:s_v * sc, rf * sc:]

                    #     out_BR, diff, score = self.model(lq_BR_rep, is_bicubic=False)
                    #     out_BR = out_BR[:, :, rf * sc:, rf * sc:]

                    #     out_L = torch.cat((out_TL, out_BL), dim=2)
                    #     out_R = torch.cat((out_TR, out_BR), dim=2)

                    #     sr_img = torch.cat((out_L, out_R), dim=3)

                    # else:
                    #     sr_img, diff, score = self.model(lr_img, is_bicubic=False)

                    sr_img, diff, score = self.model(lr_img, is_bicubic=False)

                    # inference

                    # sr = self.model(lr[:, 0, ...])
                    timer_test.hold()

                    sr_img = utility.quantize(sr_img * 255.0, self.args.rgb_range)
                    hr_img = utility.quantize(hr_img, self.args.rgb_range)

                    # metrics
                    eval_psnr += utility.calc_psnr(
                        sr_img, hr_img, scale, self.args.rgb_range,
                        benchmark=True
                    )
                    eval_ssim += utility.calc_ssim(
                        sr_img, hr_img, scale,
                        benchmark=True
                    )

                    # save results
                    if self.args.save_results:
                        filename = filename[0]
                        self.ckp.save_results_wgt(filename, sr_img, hr_img, scale)

                self.ckp.log[-1, idx_scale] = eval_psnr / len(self.loader_test1)
                self.ckp.write_log(
                    '[Epoch {}---{} x{}]\tPSNR: {:.3f} SSIM: {:.4f}'.format(
                        self.args.resume,
                        self.args.data_test1,
                        scale,
                        eval_psnr / len(self.loader_test1),
                        eval_ssim / len(self.loader_test1),
                    ))
        
        self.model.train()


    def crop_border(self, img_hr, scale):
        b, n, c, h, w = img_hr.size()

        img_hr = img_hr[:, :, :, :int(h//scale*scale), :int(w//scale*scale)]

        return img_hr

    def terminate(self):
        if self.args.test_only:
            self.test()
            return True
        else:
            epoch = self.scheduler1.last_epoch + 1
            return epoch >= self.args.epochs_encoder + self.args.epochs_sr

