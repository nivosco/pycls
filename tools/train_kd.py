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
from pycls.core.net import unwrap_model, smooth_one_hot_labels, SoftCrossEntropyLoss
from pycls.models.model_zoo import regnety, effnet
import pycls.core.checkpoint as cp
from pycls.datasets.imagenet import ImageNet
import pycls.datasets.loader as data_loader


WEIGHTS_FILE = "/tmp/pycls-download-cache/dds_baselines/160906567/RegNetY-800MF_dds_8gpu.pyth"
DATA_PATH = "/local/data/imagenet_raw/"
STUDENT = "RegNetY-800MF_W25_SE"
TEACHER = "RegNetY-800MF"
NUM_WORKERS = 8
EPOCHS = 40
WEIGHT_DECAY=0
MOMENTUM = 0.9

BASE_LEARINING_RATE = 3e-4
BATCH_SIZE = 10
STEPS = 10000 # Using total of (STEPS * BATCH_SIZE) images for training each epoch


def update_lr(optimizer, epoch, base_lr, total_epochs):
    if epoch > total_epochs * 0.75:
        new_lr = base_lr * 0.01
    elif epoch > total_epochs * 0.5:
        new_lr = base_lr * 0.1
    else:
        new_lr = base_lr

    """Sets the optimizer lr to the specified value."""
    for param_group in optimizer.param_groups:
        param_group["lr"] = new_lr
    return new_lr


def eval(model_weights, loader, replace=None, cfg=None):
    cfg_student = STUDENT if cfg is None else cfg
    print("Start evaluation on {} with weights from {}...".format(cfg_student, model_weights))
    meter = meters.TestMeter(len(loader))
    if "EfficientNet" in cfg_student:
        model = effnet(cfg_student, pretrained=False).cuda()
    else:
        model = regnety(cfg_student, pretrained=False).cuda()
    cp.load_checkpoint(model_weights, model, replace=replace)
    model.eval()
    meter.reset()
    start_time = time.time()
    for cur_iter, (inputs, labels) in enumerate(loader):
        inputs, labels = inputs.cuda(), labels.cuda(non_blocking=True)
        preds = model(inputs)
        top1_err, top5_err = meters.topk_errors(preds, labels, [1, 5])
        top1_err, top5_err = top1_err.item(), top5_err.item()
        meter.update_stats(top1_err, top5_err, inputs.size(0))
        if (cur_iter + 1) % 100 == 0:
            print("iter {}/{}".format(cur_iter + 1, len(loader)))
    print("Total evaluation time: {}s".format(round(time.time() - start_time)))
    print("**************************************")
    print("Top1 accuracy: {:.2f}".format(100 - meter.get_epoch_stats(0)["min_top1_err"]))
    print("**************************************")


def create_dataloader(data_path, split, batch):
    dataset = ImageNet(data_path, split)
    return DataLoader(dataset,
                      batch_size=batch,
                      num_workers=NUM_WORKERS,
                      pin_memory=True,
                      shuffle=(split=="train"))


def train_epoch(loader, teacher, student, loss_fun, optimizer, scaler,
                cur_epoch, kd, lr, total_epochs, batch):

    # Shuffle the data
    data_loader.shuffle(loader, cur_epoch)

    # Update the learning rate
    lr = update_lr(optimizer, cur_epoch, lr, total_epochs)
    print("Current learning rate: {}".format(lr))

    # set the models mode
    student.train()
    if kd:
        teacher.eval()

    steps = int(STEPS / (BATCH_SIZE / 10))
    start_time = time.time()
    for cur_iter, (inputs, labels) in enumerate(loader):
        # infer the models
        inputs = inputs.cuda()
        preds = student(inputs)
        if kd:
            labels = teacher(inputs)
        else:
            labels = labels.cuda(non_blocking=True)
            labels = smooth_one_hot_labels(labels)

        # calculate the loss
        loss = loss_fun(preds, labels)
        optimizer.zero_grad()
        scaler.scale(loss).backward()
        scaler.step(optimizer)
        scaler.update()
        loss = loss.item()
        if (cur_iter + 1) % 100 == 0:
            print("Loss at step {} is {:.2f}".format(cur_iter + steps
                * cur_epoch + 1, loss))
        if cur_iter + 1 >= steps:
            print("Epoch time: {}s".format(round(time.time() - start_time)))
            break


def save_ckpt(model, out=None):
    # save student weights
    checkpoint_file = 'model.pyth' if out is None else out
    checkpoint = {
        "epoch": 0,
        "model_state": unwrap_model(model).state_dict(),
    }
    torch.save(checkpoint, checkpoint_file)
    return checkpoint_file


def get_training_parameters(model):
    for name, param in model.named_parameters():
        if param.requires_grad:
            if 'bn' not in name:
                param.requires_grad = True
            else:
                param.requires_grad = False
    return model.parameters()


def train_and_eval(teacher, student, optimizer, loss_fun, dataloader_train,
        dataloader_test, kd, lr, epochs, batch, out):

    # run training
    scaler = amp.GradScaler(enabled=True)
    start_time = time.time()
    for cur_epoch in range(epochs):
        print("**************************************")
        print("Epoch {}...".format(cur_epoch))
        print("**************************************")
        # Train for one epoch
        train_epoch(dataloader_train, teacher, student, loss_fun, optimizer,
                    scaler, cur_epoch, kd, lr, epochs, batch)
    print("Total training time: {}s".format(round(time.time() - start_time)))

    # save student ckpt
    checkpoint_file = save_ckpt(student, out)
    print("Wrote checkpoint to: {}".format(checkpoint_file))

    # run evaluation
    eval(checkpoint_file, dataloader_test)


def kd_train(teacher, student, loss_fn, dataloader_train, dataloader_test,
        kd=False, lr=BASE_LEARINING_RATE, epochs=EPOCHS, batch=BATCH_SIZE,
        out=None):

    wse_parameters = get_training_parameters(student)
    optimizer = optim.SGD(wse_parameters, lr=lr, momentum=MOMENTUM, weight_decay=WEIGHT_DECAY)

    train_and_eval(teacher, student, optimizer, loss_fn, dataloader_train,
            dataloader_test, kd, lr, epochs, batch, out)



def main(weights=None, replace=None, kd=False, lr=BASE_LEARINING_RATE,
        epochs=EPOCHS, cfg=None, batch=BATCH_SIZE, eval_only=False, out=None):

    # run only evaluate if weights are provided
    if eval_only:
        print("Run Evaluation w/o Training")
        dataloader_test = create_dataloader(DATA_PATH, "val", batch)
        eval(weights, dataloader_test, replace, cfg)
        return

    # Build models
    student_cfg = STUDENT if cfg is None else cfg
    teacher = None
    if 'EfficientNet' in student_cfg:
        print("Build student model ({})".format(student_cfg))
        student = effnet(student_cfg, pretrained=False).cuda()
        if kd:
            print("Build teacher model ({})".format(TEACHER))
            teacher = effnet(TEACHER, pretrained=True).cuda()
    else:
        print("Build student model ({})".format(student_cfg))
        student = regnety(student_cfg, pretrained=False).cuda()
        if kd:
            print("Build teacher model ({})".format(TEACHER))
            teacher = regnety(TEACHER, pretrained=True).cuda()

    # load students weights with the possible weights of the teacher
    weights = weights if weights is not None else WEIGHTS_FILE
    print("Load weights from: {}".format(weights))
    cp.load_checkpoint(weights, student, replace=replace)

    # Create data loaders
    print("Create dataloaders")
    dataloader_train = create_dataloader(DATA_PATH, "train", batch)
    dataloader_test = create_dataloader(DATA_PATH, "val", batch)

    # loss
    if kd:
        print("Create L2 Loss function")
        loss_fn = lambda x, y: torch.sum(torch.pow(x - y, 2))
    else:
        print("Create Cross Entropy Loss function")
        loss_fn = SoftCrossEntropyLoss().cuda()

    # run training and evaluation after training
    print("Start training")
    kd_train(teacher, student, loss_fn, dataloader_train, dataloader_test,
             kd=kd, lr=lr, epochs=epochs, batch=batch, out=out)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", help="if provided will only run eval", default=None, type=str)
    parser.add_argument("--cfg", help="if provided will use this cfg", default=None, type=str)
    parser.add_argument("--out", help="if provided will use this filename for output", default=None, type=str)
    parser.add_argument("--replace", help="if provided will replace the string in the dict ckpt", default=None, type=str)
    parser.add_argument("--kd", help="use knowledge distillation", action="store_true")
    parser.add_argument("--eval", help="eval only", action="store_true")
    parser.add_argument("--lr", help="specifiy the learning rate", default=BASE_LEARINING_RATE, type=float)
    parser.add_argument("--epochs", help="specifiy the number of epochs", default=EPOCHS, type=int)
    parser.add_argument("--batch", help="specifiy the batch size", default=BATCH_SIZE, type=int)
    args = parser.parse_args()
    main(args.weights, args.replace, args.kd, args.lr * (args.batch / 10),
            args.epochs, args.cfg, args.batch, args.eval, args.out)
