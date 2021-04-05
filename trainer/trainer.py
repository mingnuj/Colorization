import os
import torch
import torch.nn as nn
import torch.utils.data as data
from torch.optim import lr_scheduler
from argparse import ArgumentParser
import tqdm
from trainer.utils import get_config, batch_PSNR, tensor2im
from data.data_loader import ImageLoader
from models.unet import UNet
from tensorboardX import SummaryWriter
import torchvision.utils as tvutils

parser = ArgumentParser()
parser.add_argument('--config', type=str, default='config/train_config.yaml',
                    help="training configuration")


class Trainer:
    def __init__(self, config):
        self.config = get_config(config)
        self.use_cuda = self.config['cuda']
        self.device_ids = self.config['gpu_ids']
        self.batch_size = self.config["batch_size"]
        self.model_path = self.config["load_path"]

        self.net = UNet(2, 2)

        # optimizer
        self.optimizer = torch.optim.Adam(self.net.parameters(), lr=self.config['lr'], eps=1e-08)
        # loss
        self.criterion = nn.MSELoss(reduction='sum')

        if self.use_cuda:
            self.net.to(self.device_ids[0])

    def train(self, continue_train=False):
        print("Cuda: ", torch.cuda.is_available())
        print("Device id: ", self.device_ids[0])

        if not os.path.exists(self.config["save_path"]):
            os.mkdir(self.config["save_path"])

        train_data = ImageLoader(self.config["train_data_path"], self.config["input_size"], "train", self.config["hint"])
        val_data = ImageLoader(self.config["train_data_path"], self.config["input_size"], "validation", self.config["hint"])

        train_dataloader = data.DataLoader(train_data, batch_size=self.batch_size, shuffle=True)
        val_dataloader = data.DataLoader(val_data, batch_size=self.batch_size, shuffle=True)

        if continue_train:
            print("Continue Train")
            self.net.load_state_dict(torch.load(self.model_path))

        self.net = nn.DataParallel(self.net)

        step = 0
        writer = SummaryWriter("logs")

        for epoch in range(self.config["epoch"]):
            print('Epoch {}/{}'.format(epoch + 1, self.config["epoch"]))
            print('-' * 10)

            self.net.train()

            for i, (l, gt, ab_mask) in enumerate(tqdm.tqdm(train_dataloader)):
                if self.use_cuda:
                    l = l.to(self.device_ids[0])
                    gt = gt.to(self.device_ids[0])
                    ab_mask = ab_mask.to(self.device_ids[0])
                self.optimizer.zero_grad()
                preds = self.net(ab_mask)

                loss = self.criterion(preds, gt)

                loss.backward()
                self.optimizer.step()

                self.net.eval()
                gt_image = torch.cat((l, gt), dim=1)
                pred_image = torch.cat((l, preds), dim=1)
                psnr_train = batch_PSNR(gt_image, pred_image, 1.)
                print("[epoch %d][%d/%d] loss: %.4f PSNR_train: %.4f" %
                      (epoch + 1, i + 1, len(train_dataloader), loss.item(), psnr_train))
                if step % 10 == 0:
                    # Log the scalar values
                    writer.add_scalar('loss', loss.item(), step)
                    writer.add_scalar('PSNR on training data', psnr_train, step)
                    # log the images
                    Img = tvutils.make_grid(l.data, nrow=8, normalize=True, scale_each=True)
                    Imgn = tvutils.make_grid(gt_image.data, nrow=8, normalize=True, scale_each=True)
                    Irecon = tvutils.make_grid(pred_image.data, nrow=8, normalize=True, scale_each=True)
                    writer.add_image('clean image', Img, epoch)
                    writer.add_image('noisy image', Imgn, epoch)
                    writer.add_image('reconstructed image', Irecon, epoch)
                step += 1

            torch.save(self.net.module.state_dict(), os.path.join(self.config["save_path"], "{}.tar".format(epoch)))
            print("saved at {}".format(os.path.join(self.config["save_path"], "{}.tar".format(epoch))))

            # validation part
            psnr_val = 0

            self.net.eval()
            with torch.no_grad():
                for (l, gt, ab_mask) in tqdm.tqdm(val_dataloader):
                    if self.use_cuda:
                        l = l.to(self.device_ids[0])
                        gt = gt.to(self.device_ids[0])
                        ab_mask = ab_mask.to(self.device_ids[0])
                    preds = self.net(ab_mask)

                    gt_image = torch.cat((l, gt), dim=1)
                    pred_image = torch.cat((l, preds), dim=1)
                    psnr_train = batch_PSNR(gt_image, pred_image, 1.)
                    psnr_val+=psnr_val

                psnr_val /= len(val_dataloader)
                print("\n[epoch %d] PSNR_val: %.4f" % (epoch + 1, psnr_val))
                writer.add_scalar('PSNR on validation data', psnr_val, epoch)
