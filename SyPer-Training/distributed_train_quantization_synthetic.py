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
from torch.nn import MSELoss

from periocular.backbones import iresnet, mobilefacenet
from periocular.backbones import get_model
from periocular.callbacks import LoggingCallback, VerificationCallback, CheckpointCallback
from periocular.datasets import DataLoaderX, DatasetType
from periocular.datasets import ProtocolType, PeriocularTest, PeriocularTrain, PeriocularValidation, get_file_content
from periocular.datasets.synthetic import NoIDSyntheticDataset

torch.backends.cudnn.benchmark = True

def distributed_verification_training(config):

    local_rank = config.local_rank
    
    torch.manual_seed(config.seed)
    torch.random.manual_seed(config.seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    dist.init_process_group(backend='nccl', init_method='env://')
    torch.cuda.set_device(local_rank)
    
    rank = local_rank
    world_size = dist.get_world_size()

    valset = None

    extension = ""
    if config.flip_images:
        extension = extension + "_flip"
    model_name = f"{config.base_model}_w{config.wq}a{config.aq}_{config.synth_type}{extension}"
    save_path = None
    
    cpt_model = Path(config.base_model_path).resolve()

    trainset = config.train_dataset

    train_sampler = torch.utils.data.distributed.DistributedSampler(
        trainset, shuffle=True)

    train_loader = DataLoaderX(
        local_rank=local_rank, dataset=trainset, batch_size=config.quant_batch_size,
        sampler=train_sampler, num_workers=0, pin_memory=True, drop_last=True)

    if rank == 0:
        save_path = (config.data_path / config.model_folder / model_name).resolve()
        save_path.mkdir(parents=True, exist_ok=False)
        print("Save Location:", save_path)
        print("Log Location:", save_path / f"{model_name}.log")
        print("Teacher Model Name:", config.base_model)
        print("Teacher Model Path:", config.base_model_path)
        
        # Logging
        logging.basicConfig(filename=save_path / f"{model_name}.log", level=logging.INFO, format='[%(asctime)s] %(message)s', datefmt='%Y-%m-%d %I:%M:%S %p')
        logging.info("==== Config ====")
        logging.info("Training: Quantized")
        logging.info(f"Model Name: \t{model_name}")
        for key in config.__dict__:
            logging.info(f"{key}: \t {config.__dict__[key]}")
        
        logging.info("==== \t ====")

        valset = config.test_set

    backbone = get_model(config.model, num_features=config.emb_size).to(local_rank)
    backbone.load_state_dict(torch.load(cpt_model))

    backbone_quant = get_model(config.model, num_features=config.emb_size).to(local_rank)
    backbone_quant.load_state_dict(torch.load(cpt_model))

    for ps in backbone.parameters():
        dist.broadcast(ps, 0)

    for ps in backbone_quant.parameters():
        dist.broadcast(ps, 0)


    if "resnet" in config.model:
        backbone_quant = iresnet.quantize_model(backbone_quant, config.wq, config.aq).to(local_rank)
    elif "mobilefacenet" in config.model:
        backbone_quant = mobilefacenet.quantize_model(backbone_quant, config.wq, config.aq).to(local_rank)
    else:
        raise ValueError("Unknown model given!")

    backbone = DistributedDataParallel(
        module=backbone, broadcast_buffers=False, device_ids=[local_rank])
    backbone.eval()
    
    backbone_quant = DistributedDataParallel(
        module=backbone_quant, broadcast_buffers=False, device_ids=[local_rank])
    backbone_quant.train()

    backbone_quant = iresnet.unfreeze_model(backbone_quant)

    opt_backbone = torch.optim.SGD(
        params=[{'params': backbone_quant.parameters()}],
        lr=config.quant_lr / 512 * config.quant_batch_size * world_size,
        momentum=0.9, weight_decay=config.weight_decay, nesterov=True)
    
    scheduler_backbone = torch.optim.lr_scheduler.LambdaLR(
        optimizer=opt_backbone, lr_lambda=config.lr_func)
    
    criterion = MSELoss()

    total_step = int(len(trainset) / config.quant_batch_size / world_size * config.epochs)
    if local_rank == 0: logging.info("Total Step is: %d" % total_step)

    callback_logging = LoggingCallback(config.log_interval, rank, total_step)
    callback_checkpoint = CheckpointCallback(config.save_interval, rank, total_step, model_name, save_path, quantized=True)
    callback_val = VerificationCallback(config.val_interval, rank, total_step, validation_set=valset)

    eer_min = 100
    global_step = 0
    
    for epoch in range(1, config.epochs+1):
        train_sampler.set_epoch(epoch)
        for batch_nr, (img, _) in enumerate(train_loader):
            callback_logging.start()

            global_step += 1
            img = img.cuda(local_rank, non_blocking=True)
            
            features = F.normalize(backbone_quant(img))
            
            with torch.no_grad():
                features_teacher = F.normalize(backbone(img))

            loss_v = criterion(features, features_teacher)
            loss_v.backward()

            clip_grad_norm_(backbone_quant.parameters(), max_norm=5, norm_type=2)

            opt_backbone.step()
            opt_backbone.zero_grad()
        
            callback_logging(global_step, loss_v.item())
        
        
        if rank == 0 and epoch > config.epochs-2:
            logging.info(f"Epoch: {epoch} => Saving models at time step {global_step}.")
            callback_checkpoint(global_step, backbone_quant, None, force=True)

        if epoch > config.quant_qat_epochs:
            backbone_quant = iresnet.freeze_model(backbone_quant)

        if rank == 0 and (epoch == 4 or epoch == 6):#rank == 0 and epoch % 5 == 0:
            metrics = callback_val(global_step, backbone_quant, force=True)
            if metrics is not None and metrics.eer < eer_min:
                print("Tested model is new best. Saving model.")
                logging.info(f"[Step {global_step}] Current model is new best model. Saving model.")
                callback_checkpoint(global_step, backbone_quant, None, force=True, include_step=False)
                eer_min = metrics.eer

        scheduler_backbone.step()
        

    if rank == 0:
        callback_checkpoint(global_step, backbone_quant, None, force=True)

    dist.destroy_process_group()


if __name__ == "__main__":

    config_file = [f for f in sys.argv if "pkl" in f][0]
    
    with open(f"{config_file}", "rb") as f:
        config = pickle.load(f)

    protocol = {"open_world_valopen":ProtocolType.OPEN_WORLD_OPEN_VAL, 
                "closed_world":ProtocolType.CLOSED_WORLD,
                "open_world_valclosed":ProtocolType.OPEN_WORLD_CLOSED_VAL}[config.protocol]
    
    
    if      config.train_dataset_type == "NoIDSyntheticDataset":
        config.train_dataset = NoIDSyntheticDataset(flip_L_to_R=config.flip_images,
                                             dataset_root=config.train_dataset_path)
    elif    config.train_dataset_type == "UFPR":
        content = get_file_content("/data/jkolf/datasets/UFPR-Periocular/", protocol, DatasetType.TRAIN, config.fold)
        config.train_dataset = PeriocularTrain(content, flip_L_to_R=config.flip_images, 
                                                 dataset_root=config.train_dataset_path)
    else:
        raise ValueError("Unknown dataset type given!")
    
    config.local_rank   = int(os.environ["LOCAL_RANK"])
    config.data_path = Path(config.data_path).resolve()
    
    if config.local_rank == 0:
        val_content = get_file_content("/data/jkolf/datasets/UFPR-Periocular/", protocol, DatasetType.VAL, config.fold)
        config.test_set = PeriocularValidation(val_content, amount_genuine=20, amount_imposter=1000, flip_L_to_R=config.flip_images)
    else:
        config.test_set = None
    
    def lr_step_func(epoch):
        return 0.1 ** len(
            [m for m in [8] if m - 1 <= epoch])  # [8, 14,20,25] [m for m in [10,20,25,30,35]
    config.lr_func = lr_step_func

    try:
        distributed_verification_training(config)
    except KeyboardInterrupt:
        logging.info("Ctrl+C User Input.")
    except Exception as e:
        logging.exception(e)
        traceback.print_exc()



