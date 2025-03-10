import argparse
from distutils.log import Log
import logging
import os
import time
from types import SimpleNamespace
from pathlib import Path
import sys
import pickle
import traceback

import torch
import torch.distributed as dist
import torch.nn.functional as F
from torch.nn.parallel.distributed import DistributedDataParallel
import torch.utils.data.distributed
from torch.nn.utils import clip_grad_norm_
from torch.nn import CrossEntropyLoss

from periocular.backbones import get_model
from periocular.callbacks import LoggingCallback, VerificationCallback, CheckpointCallback
from periocular.datasets import DataLoaderX, PeriocularTest, PeriocularTrain, PeriocularValidation
from periocular.datasets import ProtocolType, DatasetType, get_file_content
from periocular.datasets import NIRTest, NIRTrain
from periocular.losses import ArcFace


torch.backends.cudnn.benchmark = True

def distributed_verification_training(config):

    local_rank = config.local_rank
    
    torch.manual_seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    
    rank = local_rank#dist.get_rank()
    world_size = dist.get_world_size()

    trainset =  config.train_set
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=config.batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    valset = None

    extension = ""
    if config.flip_images:
        extension = extension + "_flip"

    model_name = f"{config.model}_f{config.fold}_m{config.m}_s{config.s}_emb{config.emb_size}_lr{config.base_lr}_b{config.batch_size}{extension}"
    save_path = None
    if rank == 0:
        save_path = (config.data_path / "models" / model_name).resolve()
        save_path.mkdir(parents=True, exist_ok=False)
        
        with open(save_path / "config.pkl", "wb") as f:
            pickle.dump(config, f)
        
        print("Save Location:", save_path)
        logging.basicConfig(filename=save_path / f"{model_name}.log", level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
        print("Log Location:", save_path / f"{model_name}.log")
        valset = config.test_set

        logging.info("==== Config ====")
        logging.info("Training: Non-Quantized")
        logging.info(f"Seed:\t{config.seed}")
        logging.info(f"Batch-Size:\t{config.batch_size}")
        logging.info(f"Epochs:\t{config.epochs}")
        logging.info(f"Shuffle:\t{config.shuffle}")
        logging.info(f"Emb-Size:\t{config.emb_size}")
        logging.info(f"Margin:\t{config.m}")
        logging.info(f"Scale:\t{config.s}")
        logging.info(f"Base LR:\t{config.base_lr}")
        logging.info(f"Weight Decay:\t{config.weight_decay}")
        logging.info(f"Fold:\t{config.fold}")
        logging.info(f"Model:\t{config.model}")
        logging.info(f"Flipping:\t{config.flip_images}")
        logging.info(f"Img-Size:\t{config.img_size}")
        logging.info(f"Data-Path:\t{config.data_path}")
        logging.info(f"Protocol:\t{config.protocol}")
        logging.info(f"Log-Interval:\t{config.log_interval}")
        logging.info(f"Save-Interval:\t{config.save_interval}")
        logging.info(f"Val-Interval:\t{config.val_interval}")
        logging.info(f"WQ:\t{config.wq}")
        logging.info(f"AQ:\t{config.aq}")
        logging.info(f"Base-Mdl:\t{config.base_model}")
        logging.info(f"Train-Set #Classes:\t{trainset.num_classes}")
        logging.info("==== \t ====")


    backbone = get_model(config.model, num_features=config.emb_size).to(local_rank)
    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.train()

    header = ArcFace(
                    in_features=config.emb_size, 
                    out_features=trainset.num_classes, 
                    m=config.m,
                    s=config.s
                ).to(local_rank)
    
    header = DistributedDataParallel(
        module=header, broadcast_buffers=False, device_ids=[local_rank])
    header.train()

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone.parameters()}],
        lr=config.base_lr / 512 * config.batch_size * world_size,
        momentum=0.9, weight_decay=config.weight_decay)
    opt_header = torch.optim.SGD(
        params=[{'params': header.parameters()}],
        lr=config.base_lr / 512 * config.batch_size * world_size,
        momentum=0.9, weight_decay=config.weight_decay)

    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=config.lr_func)
    scheduler_header = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_header, lr_lambda=config.lr_func)        

    criterion = CrossEntropyLoss()

    total_step = int(len(trainset) / config.batch_size / world_size * config.epochs)
    if local_rank == 0: logging.info("Total Step is: %d" % total_step)

    callback_logging = LoggingCallback(config.log_interval, rank, total_step)
    callback_checkpoint = CheckpointCallback(config.save_interval, rank, total_step, model_name, save_path)
    callback_val = VerificationCallback(config.val_interval, rank, total_step, validation_set=valset)

    eer_min = 100

    global_step = 0
    for epoch in range(1, config.epochs+1):
        train_sampler.set_epoch(epoch)
        for _, (img, label) in enumerate(train_loader):
            callback_logging.start()

            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            label = label.cuda(local_rank, non_blocking=True)

            features = F.normalize(backbone(img))

            thetas = header(features, label)
            loss_v = criterion(thetas, label)
            loss_v.backward()

            clip_grad_norm_(backbone.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_header.step()

            opt_backbone.zero_grad()
            opt_header.zero_grad()

            callback_logging(global_step, loss_v.item())
            #callback_checkpoint(global_step, backbone, header)
            #callback_val(global_step, backbone)

        if rank == 0: #and epoch > config.epochs-2:
            if local_rank == 0: logging.info(f"Epoch: {epoch} => Saving models at time step {global_step}.")
            callback_checkpoint(global_step, backbone, header, force=True)

        if rank == 0 and epoch % config.param_test_freq == 0 and epoch >= config.param_min_test_epoch:
            metrics = callback_val(global_step, backbone, force=True)
            if metrics is not None and metrics.eer < eer_min:
                print("Tested model is new best. Saving model.")
                logging.info(f"[Step {global_step}] Current model is new best model. Saving model.")
                callback_checkpoint(global_step, backbone, header, force=True, include_step=False)
                eer_min = metrics.eer
                with open(save_path / "best_metrics.pkl", "wb") as f:
                    pickle.dump(metrics, f)
                
        scheduler_backbone.step()
        scheduler_header.step()
    
    if rank == 0:
        callback_checkpoint(global_step, backbone, header, force=True)
    
    dist.destroy_process_group()


if __name__ == "__main__":

    config_file = [f for f in sys.argv if "pkl" in f][0]
    
    with open(f"{config_file}", "rb") as f:
        config = pickle.load(f)

    if config.ufpr_path is None:
        config.train_set = NIRTrain(
                                    img_size=config.img_size,
                                    flip_L_to_R=config.flip_images,
                                    dataset_root=config.train_dataset_path
                                    )
                    
        config.test_set = NIRTest(
                                    img_size=config.img_size,
                                    flip_L_to_R=config.flip_images,
                                    dataset_root=config.train_dataset_path
                                 )
    else:
        raise ValueError("Could not load NIR.")
    
    config.local_rank   = int(os.environ["LOCAL_RANK"])
    config.data_path = Path(config.data_path).resolve()
    
    def lr_step_func(epoch):
        return ((epoch + 1) / (4 + 1)) ** 2 if epoch < -1 else 0.1 ** len(
            [m for m in [8,15,25] if m - 1 <= epoch])  # [m for m in [10,20,25,30,35]
    config.lr_func = lr_step_func

    if config.local_rank == 0:
        print("===== Params =====")
        print(f" SEED = \t{config.seed}")
        print(f" MODEL = \t{config.model}")
        print(f" BATCH_SIZE = \t{config.batch_size}")
        print(f" EPOCHS = \t{config.epochs}")
        print(f" EMB_SIZE = \t{config.emb_size}")
        print(f" ARCFACE_M = \t{config.m}")
        print(f" ARCFACE_S = \t{config.s}")
        print(f" BASE_LR = \t{config.base_lr}")
        print(f" FOLD = \t{config.fold}")
        print(f" IMG-SIZE = \t{config.img_size}")
        print(f" PROTOCL = \t{config.protocol}")
        print(f" LOG INT = \t{config.log_interval}")
        print(f" SAVE INT = \t{config.save_interval}")
        print(f" VAL INT = \t{config.val_interval}")
        print(f" BASE LR = \t{config.base_lr}")
        print(f" FLIP L2R = \t{config.flip_images}")

    try:
        distributed_verification_training(config)
    except KeyboardInterrupt:
        logging.info("Ctrl+C User Input.")
    except Exception as e:
        logging.exception(e)
        traceback.print_exc()