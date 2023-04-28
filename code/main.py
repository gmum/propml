import os
import random
import sys

import numpy as np
import torch
import torch.nn.parallel
import torch.optim
import torch.utils.data.distributed
import torchvision.transforms as transforms
from randaugment import RandAugment
from torch.cuda.amp import GradScaler, autocast
from torch.distributions import Gumbel
from torch.optim import lr_scheduler
from torchmetrics.functional.classification import multilabel_precision, multilabel_recall

import wandb
from local_settings import PRETRAINED_PATHS, DATASET_PATHS
from src.helper_functions.helper_functions import CutoutPIL, ModelEma, add_weight_decay, mAP
from src.models import create_model
from utils_algo import PMLL, precision_recall_f1, precision_recall_f1_scikit
from utils_data import get_VOC2007, generate_noisy_labels, VOC2007_handler, VOC2007_handler_aug, CocoDetection

DATASET_CONFIG = {
    'voc': {
        'image_size': 224,
        'n_classes': 20
    },
    'coco': {
        'image_size': 224,
        'n_classes': 80
    }
}

N_WORKERS = 8  # max(1, multiprocessing.cpu_count() - 1)
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def validate_multi(val_loader, model, ema_model):
    print("starting validation")
    Sig = torch.nn.Sigmoid()
    preds_regular = []
    preds_ema = []
    targets = []
    for i, (input, target, _) in enumerate(val_loader):
        input = input.to(DEVICE)
        target = target.to(DEVICE)
        # target = target.max(dim=1)[0]
        # compute output
        with torch.no_grad():
            with autocast():
                output_regular = Sig(model(input))
                output_ema = Sig(ema_model.module(input))

        # for mAP calculation
        preds_regular.append(output_regular.detach())
        preds_ema.append(output_ema.detach())
        targets.append(target.detach())

    preds_regular = torch.cat(preds_regular, dim=0)
    confidences_regular = Sig(preds_regular)
    preds_ema = torch.cat(preds_ema, dim=0)
    confidences_ema = Sig(preds_ema)
    targets = torch.cat(targets, dim=0)

    mAP_score_regular = mAP(targets.cpu().detach().numpy(), preds_regular.cpu().detach().numpy())
    mAP_score_ema = mAP(targets.cpu().detach().numpy(), preds_ema.cpu().detach().numpy())
    print("mAP score regular {:.2f}, mAP score EMA {:.2f}".format(mAP_score_regular, mAP_score_ema))
    OF1, CF1 = precision_recall_f1(targets.cpu().detach().numpy(),
                                   confidences_regular.cpu().detach().numpy())
    OF1_scikit, CF1_scikit = precision_recall_f1_scikit(targets.cpu().detach().numpy(),
                                                        confidences_regular.cpu().detach().numpy())
    print('CF1: {:.2f} '.format(CF1),
          'OF1: {:.2f} '.format(OF1))
    OF1_ema, CF1_ema = precision_recall_f1(targets.cpu().detach().numpy(),
                                           confidences_ema.cpu().detach().numpy())
    OF1_ema_scikit, CF1_ema_scikit = precision_recall_f1_scikit(targets.cpu().detach().numpy(),
                                                                confidences_ema.cpu().detach().numpy())
    print('EMA CF1: {:.2f} '.format(CF1_ema),
          'EMA OF1: {:.2f} '.format(OF1_ema))

    prec = multilabel_precision(preds_regular.cpu().detach(), targets.cpu().detach(),
                                num_labels=wandb.config.n_classes, average='macro')
    recall = multilabel_recall(preds_regular.cpu().detach(), targets.cpu().detach(),
                               num_labels=wandb.config.n_classes, average='macro')
    return mAP_score_regular, mAP_score_ema, CF1, OF1, CF1_ema, OF1_ema, OF1_scikit, CF1_scikit, OF1_ema_scikit, \
        CF1_ema_scikit, prec, recall


def train(model, train_loader, val_loader, model_save_path):
    ema = ModelEma(model, 0.9997)

    parameters = add_weight_decay(model, wandb.config.weight_decay)
    optimizer = torch.optim.Adam(params=parameters, lr=wandb.config.lr, weight_decay=0)  # true wd, filter_bias_and_bn
    steps_per_epoch = len(train_loader)
    if wandb.config.use_scheduler:
        scheduler = lr_scheduler.OneCycleLR(optimizer, max_lr=wandb.config.lr, steps_per_epoch=steps_per_epoch,
                                            epochs=wandb.config.epochs,
                                            pct_start=0.2)

    mAP_bar = 0.0
    ema_mAP_bar = 0.0
    trainInfoList = []
    scaler = GradScaler()
    Sig = torch.nn.Sigmoid()

    if wandb.config.use_noise:
        noise = Gumbel(
            torch.tensor(0.0, device=DEVICE), torch.tensor(1.0, device=DEVICE)
        )
        max_iter = wandb.config.epochs * len(train_loader) * 0.7
        print('\033[0;1;31mUse noise in outputs\033[0m')

    loss_fn = PMLL(reduce='mean', alpha=wandb.config.pmll_a)

    alpha_noise_log = 0

    for epoch in range(wandb.config.epochs):
        model.train()
        losses = 0
        preds_regular = []
        targets = []

        iteration = epoch * len(train_loader)
        for i, (img, target, ind) in enumerate(train_loader):
            img = img.to(DEVICE)
            target = target.to(DEVICE)

            with autocast():  # mixed precision
                output = model(img).float()  # sigmoid will be done in loss !
                output_regular = Sig(output)

            if wandb.config.use_noise:
                alpha_noise = max(0.0, 1 - iteration / max_iter)
                alpha_noise_log += alpha_noise
                output = output + alpha_noise * (
                        noise.sample(output.shape) - noise.sample(output.shape)
                )

            loss, first_part, second_part = loss_fn(output, target)
            losses += loss.item()
            wandb.log({'train/iterloss': loss, 'train/first_part': first_part, 'train/second_part': second_part})

            preds_regular.append(output_regular.detach())
            targets.append(target.detach())

            model.zero_grad()

            scaler.scale(loss).backward()
            # loss.backward()

            scaler.step(optimizer)
            scaler.update()
            # optimizer.step()

            if wandb.config.use_scheduler:
                scheduler.step()

            ema.update(model)
            # store information
            if i % 20 == 0:
                trainInfoList.append([epoch, i, loss.item()])
                print('Epoch [{}/{}], Step [{}/{}], Loss: {:.1f}'
                      .format(epoch, wandb.config.epochs, str(i).zfill(3), str(steps_per_epoch).zfill(3),
                              loss.item()))

        preds_regular = torch.cat(preds_regular, dim=0)
        targets = torch.cat(targets, dim=0)
        mAP_score_regular = mAP(targets.cpu().detach().numpy(), preds_regular.cpu().detach().numpy())
        prec = multilabel_precision(preds_regular.cpu().detach(), targets.cpu().detach(),
                                    num_labels=wandb.config.n_classes, average='macro')
        recall = multilabel_recall(preds_regular.cpu().detach(), targets.cpu().detach(),
                                   num_labels=wandb.config.n_classes, average='macro')
        wandb.log({
            'train/loss': losses / len(train_loader),
            'train/alpha_noise': alpha_noise_log / len(train_loader),
            'train/mAP': mAP_score_regular,
            'train/precision': prec,
            'train/recall': recall
        })

        model.eval()
        mAP_score, mAP_score_ema, CF1, OF1, CF1_ema, OF1_ema, OF1_scikit, CF1_scikit, OF1_ema_scikit, CF1_ema_scikit, \
            prec, recall = validate_multi(val_loader, model, ema)
        model.train()

        if mAP_score > mAP_bar:
            mAP_bar = mAP_score
            torch.save(model.state_dict(), os.path.join(model_save_path, 'best.pth'))

        if mAP_score_ema > ema_mAP_bar:
            ema_mAP_bar = mAP_score_ema
            torch.save(ema.module.state_dict(), os.path.join(model_save_path, 'best_ema.pth'))

        print('current_mAP = {:.2f}, highest_mAP = {:.2f}\n'.format(max(mAP_score, mAP_score_ema),
                                                                    max(mAP_bar, ema_mAP_bar)))
        wandb.log({
            "test/mAP": mAP_score,
            "test/mAP_ema": mAP_score_ema,
            'test/CF1': CF1,
            'test/OF1': OF1,
            'test/CF1_ema': CF1_ema,
            'test/OF1_ema': OF1_ema,
            'test/CF1_scikit': CF1_scikit,
            'test/OF1_scikit': OF1_scikit,
            'test/OF1_ema_scikit': OF1_ema_scikit,
            'test/CF1_ema_scikit': CF1_ema_scikit,
            'test/precision': prec,
            'test/recall': recall,
            'test/mAP_best': max(mAP_bar, ema_mAP_bar)
        })


def main():
    config = {
        "dataset": 'coco',
        "q": 0.2,
        "lr": 0.00007413412586892413,
        "weight_decay": 6.627112148968199e-7,
        "model": "tresnet_l",
        "epochs": 50,
        "batch_size": 64,
        "use_noise": False,
        "use_scheduler": True,
        "pmll_a": 0.05999794429127484
    }

    os.makedirs('/tmp/ap', exist_ok=True)
    wandb.init(project="pml", config=config, reinit=True)

    wandb.config.update(DATASET_CONFIG[wandb.config.dataset])

    seed = 1
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True

    model_save_path = os.path.join('models', wandb.run.name)
    os.makedirs(model_save_path, exist_ok=True)

    # Setup model
    print('creating model...')
    model = create_model(wandb.config)
    model = model.to(DEVICE)

    if PRETRAINED_PATHS.get(wandb.config.model):  # make sure to load pretrained ImageNet model
        state = torch.load(PRETRAINED_PATHS[wandb.config.model], map_location='cpu')
        filtered_dict = {k: v for k, v in state['model'].items() if
                         (k in model.state_dict() and 'head.fc' not in k)}
        model.load_state_dict(filtered_dict, strict=False)
        print('loaded pretrained model', wandb.config.model, file=sys.stderr)

    print('init done', file=sys.stderr)

    if wandb.config.dataset == 'voc':
        # VOC Data loading
        train_transform_aug = transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize((wandb.config.image_size, wandb.config.image_size)),
            CutoutPIL(cutout_factor=0.5),
            RandAugment(),
            transforms.ToTensor()])

        test_transform = transforms.Compose([
            transforms.Resize((wandb.config.image_size, wandb.config.image_size)),
            transforms.ToTensor()])

        data_path_val = f'{DATASET_PATHS[wandb.config.dataset]}/VOCdevkit/VOC2007'
        data_path_train = f'{DATASET_PATHS[wandb.config.dataset]}/VOCdevkit/VOC2007'

        train_images, train_labels, test_images, test_labels = get_VOC2007(data_path_train, data_path_val)
        train_noisy_labels = generate_noisy_labels(train_labels, noise_rate=wandb.config.q)
        train_dataset = VOC2007_handler_aug(train_images, train_noisy_labels, data_path_train,
                                            transform_aug=train_transform_aug)
        test_dataset = VOC2007_handler(test_images, test_labels, data_path_val, transform=test_transform)
    elif wandb.config.dataset == 'coco':
        # COCO Data loading
        instances_path_val = os.path.join(DATASET_PATHS[wandb.config.dataset], 'annotations/instances_val2014.json')
        instances_path_train = os.path.join(DATASET_PATHS[wandb.config.dataset], 'annotations/instances_train2014.json')

        data_path_val = f'{DATASET_PATHS[wandb.config.dataset]}/val2014'  # args.data
        data_path_train = f'{DATASET_PATHS[wandb.config.dataset]}/train2014'  # args.data

        test_dataset = CocoDetection(data_path_val,
                                     instances_path_val,
                                     transforms.Compose([
                                         transforms.Resize((wandb.config.image_size, wandb.config.image_size)),
                                         transforms.ToTensor(),
                                         # normalize, # no need, toTensor does normalization
                                     ]))
        train_dataset = CocoDetection(data_path_train,
                                      instances_path_train,
                                      transforms.Compose([
                                          transforms.RandomHorizontalFlip(),
                                          transforms.Resize((wandb.config.image_size, wandb.config.image_size)),
                                          CutoutPIL(cutout_factor=0.5),
                                          RandAugment(),
                                          transforms.ToTensor(),
                                          # normalize,
                                      ]), noise=wandb.config.q)

    else:
        raise NotImplemented

    print("len(val_dataset): ", len(test_dataset))
    print("len(train_dataset): ", len(train_dataset))

    # Data loading
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=wandb.config.batch_size, shuffle=True,
        num_workers=N_WORKERS, pin_memory=True)

    test_loader = torch.utils.data.DataLoader(
        test_dataset, batch_size=wandb.config.batch_size, shuffle=False,
        num_workers=N_WORKERS, pin_memory=False)

    train(model, train_loader, test_loader, model_save_path)


if __name__ == '__main__':

    if len(sys.argv) > 1:
        sweep_id = sys.argv[1]

        if sweep_id == 'new':
            print(wandb.sweep({
                'metric': {
                    'name': 'test/mAP',
                    'goal': 'maximize',
                },
                'early_terminate': {
                    'min_iter': 8,
                    'type': 'hyperband'
                },
                'method': 'bayes',
                'parameters': {
                    'dataset': {
                        'value': 'coco'
                    },
                    'q': {
                        'values': [0.1],
                    },
                    'weight_decay': {
                        'distribution': 'log_uniform_values',
                        'max': 0.0001,
                        'min': 0.000001,
                    },
                    'use_scheduler': {
                        'distribution': 'categorical',
                        'values': [False, True]
                    },
                    'pmll_a': {
                        'distribution': 'log_uniform_values',
                        'max': 0.01,
                        'min': 0.001,
                    },

                }}, project="pml"))
        else:
            wandb.agent(sweep_id=sweep_id, function=main)
    else:
        main()
