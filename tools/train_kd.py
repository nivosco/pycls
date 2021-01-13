#!/usr/bin/env python3

# Copyright (c) Facebook, Inc. and its affiliates.
#
# This source code is licensed under the MIT license found in the
# LICENSE file in the root directory of this source tree.

"""Train with knowledge distialation classification model."""

import argparse
import time
import torch
import torch.optim as optim
from torch.utils.data import DataLoader
import torch.cuda.amp as amp
import pycls.core.meters as meters
from pycls.core.net import unwrap_model
from pycls.models.model_zoo import regnety
import pycls.core.checkpoint as cp
from pycls.datasets.imagenet import ImageNet
import pycls.datasets.loader as data_loader


WEIGHTS_FILE = "/tmp/pycls-download-cache/dds_baselines/160906567/RegNetY-800MF_dds_8gpu.pyth"
DATA_PATH = "/local/data/imagenet_raw/"
STUDENT = "RegNetY-800MF_W25_SE"
TEACHER = "RegNetY-800MF"
NUM_WORKERS = 8
EPOCHS = 50
BASE_LEARINING_RATE = 1e-4
BATCH_SIZE = 64
STEPS = 300 # Using total of (STEPS * BATCH_SIZE) images for training each epoch


def update_lr(optimizer, epoch):
    if epoch > EPOCHS * 0.8:
        new_lr = BASE_LEARINING_RATE * 0.01
    elif epoch > EPOCHS * 0.6:
        new_lr = BASE_LEARINING_RATE * 0.1
    else:
        new_lr = BASE_LEARINING_RATE

    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def eval(model_weights, loader):
    print("Start evaluation...")
    meter = meters.TestMeter(len(loader))
    model = regnety(STUDENT, pretrained=False).cuda()
    cp.load_checkpoint(model_weights, model)
    model.eval()
    meter.reset()
    start_time = time.time()
    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.update_stats(top1_err, top5_err, inputs.size(0))
        print("iter {}/{}".format(cur_iter + 1, len(loader)))
    print("Total evaluation time: {}s".format(round(time.time() - start_time)))
    print("**************************************")
    print("Top1 accuracy: {}".format(100 - meter.get_epoch_stats(0)["min_top1_err"]))
    print("**************************************")


def create_dataloader(data_path, split):
    dataset = ImageNet(data_path, split)
    return DataLoader(dataset,
                      batch_size=BATCH_SIZE,
                      num_workers=NUM_WORKERS,
                      pin_memory=True,
                      shuffle=split == "train")


def train_epoch(loader, teacher, student, loss_fun, optimizer, scaler, cur_epoch):

    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)

    # Update the learning rate
    lr = update_lr(optimizer, cur_epoch)
    print("Current learning rate: {}".format(lr))

    # set the models mode
    student.train()
    teacher.eval()

    start_time = time.time()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # infer the models
        inputs = inputs.cuda()
        preds = student(inputs)
        labels = teacher(inputs)

        # calculate the loss
        loss = loss_fun(preds, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss = loss.item()
        print("Loss at step {} is {:.2f}".format(cur_iter + STEPS * cur_epoch, loss))
        if cur_iter + 1 >= STEPS:
            print("Epoch time: {}s".format(round(time.time() - start_time)))
            break


def save_ckpt(model):
    # save student weights
    checkpoint_file = 'model.pyth'
    checkpoint = {
        "epoch": 0,
        "model_state": unwrap_model(model).state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def get_training_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad and 'w_se' not in name:
            param.requires_grad = False
    return model.parameters()


def train_and_eval(teacher, student, optimizer, loss_fun, dataloader_train, dataloader_test, run_eval):

    # run training
    scaler = amp.GradScaler(enabled=True)
    start_time = time.time()
    for cur_epoch in range(EPOCHS):
        print("**************************************")
        print("Epoch {}...".format(cur_epoch))
        print("**************************************")
        # Train for one epoch
        train_epoch(dataloader_train, teacher, student, loss_fun, optimizer, scaler, cur_epoch)
    print("Total training time: {}s".format(round(time.time() - start_time)))

    # save student ckpt
    checkpoint_file = save_ckpt(student)
    print("Wrote checkpoint to: {}".format(checkpoint_file))

    # run evaluation
    if run_eval:
        eval(checkpoint_file, dataloader_test)


def kd_train(teacher, student, loss_fn, dataloader_train, dataloader_test, only_se=False, run_eval=False):

    if only_se:
        wse_parameters = get_training_parameters(student)
        optimizer = optim.Adam(wse_parameters, lr=BASE_LEARINING_RATE)
    else:
        optimizer = optim.Adam(student.parameters(), lr=BASE_LEARINING_RATE)

    train_and_eval(teacher, student, optimizer, loss_fn, dataloader_train, dataloader_test, run_eval)



def main(weights=None):

    # run only evaluate if weights are provided
    if weights:
        dataloader_test = create_dataloader(DATA_PATH, "val")
        eval(weights, dataloader_test)
        return

    # Build models
    teacher = regnety(TEACHER, pretrained=True).cuda()
    student = regnety(STUDENT, pretrained=False).cuda()

    # load students weights with the possible weights of the teacher
    cp.load_checkpoint(WEIGHTS_FILE, student, strict=False)

    # Create data loaders
    dataloader_train = create_dataloader(DATA_PATH, "train")
    dataloader_test = create_dataloader(DATA_PATH, "val")

    # loss
    loss_fn = lambda x, y: torch.sum(torch.pow(x - y, 2))

    # run training and evaluation after training
    kd_train(teacher, student, loss_fn, dataloader_train, dataloader_test, only_se=True)
    kd_train(teacher, student, loss_fn, dataloader_train, dataloader_test, run_eval=True)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="if provided will only run eval", default=None, type=str)
    args = parser.parse_args()
    main(args.weights)
