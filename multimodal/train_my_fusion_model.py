import os
from os.path import join as pjoin
from tqdm import tqdm
from datetime import datetime
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader

from dataloaders.MultimodalManipulationDataset import MyMultimodalManipulationDataset
from models.sensor_fusion import MySensorFusionSelfSupervised

import gc
gc.collect()

NUM_EPOCH = 50
NUM_BATCH = 128
NUM_WORKERS = 8
INIT_LR = 0.003
WEIGHT_DECAY = 0.0005

DATA_ROOT_PATH = "/home/jb/projects/Code/IERG5350/project/ierg5350_rl_course_project/multimodal/dataset/simulation_data"
SAVE_CKPT_DIR = "runs"

os.environ["CUDA_VISIBLE_DEVICES"] = "1"


def check_data(input):
    print("[{}]: {}, {}, {}".format("input", type(input), input.shape, input.dtype))


def train(train_dataloader, train_model, optimizer, train_criterion, device="cpu"):
    train_model.train()
    loss_sum = 0.0

    for iter, sample in enumerate(tqdm(train_dataloader)):
        inputs = [sample["color_prev"].to(device),
                  sample["depth_prev"].to(device),
                  sample["ft_prev"].to(device),
                  sample["action"].to(device)]

        color_gt = sample["color"].transpose(1, 3).transpose(2, 3).to(device)
        depth_gt = sample["depth"].transpose(1, 3).transpose(2, 3).to(device)
        contact_gt = sample["contact"].view(-1, 1, 1).to(device)

        optimizer.zero_grad()

        color_pred, depth_pred, contact_pred = train_model(*inputs)
        # color_pred, depth_pred, contact_pred = train_model(
        #     sample["color_prev"].to(device),
        #     sample["depth_prev"].to(device),
        #     sample["ft_prev"].to(device),
        #     sample["action"].to(device))

        loss = train_criterion["image"](color_pred, color_gt) + \
               train_criterion["image"](depth_pred, depth_gt) + \
               100 * train_criterion["label"](contact_pred, contact_gt)

        loss.backward()
        optimizer.step()

        loss_sum += loss.item()

        # if iter > 2:
        #     break
    return loss_sum / len(train_dataloader)


def evaluate(eval_dataloader, eval_model, eval_criterion):
    model.eval()
    loss_sum = 0.0

    for _, sample in enumerate(tqdm(eval_dataloader)):
        inputs = [sample["color_prev"].to(device),
                  sample["depth_prev"].to(device),
                  sample["ft_prev"].to(device),
                  sample["action"].to(device)]

        color_gt = sample["color"].transpose(1, 3).transpose(2, 3).to(device)
        depth_gt = sample["depth"].transpose(1, 3).transpose(2, 3).to(device)
        contact_gt = sample["contact"].view(-1, 1, 1).to(device)

        with torch.no_grad():
            color_pred, depth_pred, contact_pred = eval_model(*inputs)

        loss = eval_criterion["image"](color_pred.double(), color_gt.double()) + \
               eval_criterion["image"](depth_pred.double(), depth_gt.double()) + \
               100 * eval_criterion["label"](contact_pred.double(), contact_gt.double())

        loss_sum += loss.item()

    return loss_sum / len(eval_dataloader)


if __name__ == "__main__":
    # init device
    device = "cuda" if torch.cuda.is_available() else torch.device('cpu')
    print("[info]: training using {}".format(device))

    date = datetime.now()
    ckpt_dir = pjoin(SAVE_CKPT_DIR, date.strftime("%Y-%m-%d-%H-%M"))
    os.makedirs(ckpt_dir)

    # init dataset
    train_set = MyMultimodalManipulationDataset(pjoin(DATA_ROOT_PATH, "train"), device)
    eval_set = MyMultimodalManipulationDataset(pjoin(DATA_ROOT_PATH, "eval"), device)

    train_loader = DataLoader(train_set, batch_size=NUM_BATCH, shuffle=True, num_workers=NUM_WORKERS)
    eval_loader = DataLoader(eval_set, batch_size=NUM_BATCH, shuffle=False, num_workers=NUM_WORKERS)

    # init model
    model = MySensorFusionSelfSupervised(device).to(device)

    # init optimizer and criterion
    optimizer = torch.optim.Adam(model.parameters(),
                                 lr=INIT_LR,
                                 betas=(0.9, 0.999),
                                 weight_decay=WEIGHT_DECAY)
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=5, gamma=0.5)
    criterion = {
        "image": F.mse_loss,
        "label": F.binary_cross_entropy_with_logits
    }

    eval_loss_prev = None
    best_epoch = 0
    for i in range(NUM_EPOCH):
        print("\n\n[info] Starting the epoch {}".format(i))
        # train
        train_loss = train(train_loader, model, optimizer, criterion, device)
        scheduler.step(i)
        # print train loss
        print("\n[info]: Trained {} epochs, loss: {:.6f}".format(i, train_loss))

        # eval
        eval_loss = evaluate(eval_loader, model, criterion)
        # print eval loss
        print("\n[info]: Evaluated {} epochs, loss: {:.6f}".format(i, eval_loss))

        # compare eval loss to save the model
        if not eval_loss_prev:
            # save the eval loss for the first evaluation
            eval_loss_prev = eval_loss
        else:
            # compare the last loss with current loss, save the model if the result is better
            if eval_loss_prev > eval_loss:
                eval_loss_prev = eval_loss
                best_epoch = i

                # save the model
                print("[info]: better loss found ")
                torch.save(model.state_dict(), pjoin(ckpt_dir, "ckpt_full_best.pth"))
                torch.save(model.obs_encoder.state_dict(), pjoin(ckpt_dir, "ckpt_encoder_best.pth"))

    # save the final model and save the encoder only
    torch.save(model.state_dict(), pjoin(ckpt_dir, "ckpt_full_final.pth"))
    torch.save(model.obs_encoder.state_dict(), pjoin(ckpt_dir, "ckpt_encoder_final.pth"))
    with open(pjoin(ckpt_dir, "best_pred.txt"), 'wt') as f:
        f.write("best checkpoint: {}\n".format(best_epoch))
        f.write("best loss: {:.6f}".format(eval_loss_prev))
