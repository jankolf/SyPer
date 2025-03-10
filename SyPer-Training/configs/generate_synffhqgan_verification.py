from types import SimpleNamespace
import pickle
from pathlib import Path

if __name__ == "__main__":

    config = SimpleNamespace()

    # Settings Train Dataset
    config.train_dataset_type = "NoIDSyntheticDataset"
    config.fold = 1 # 1,2,3
    config.protocol = "open_world_valclosed"
    config.img_size     = 224
    config.flip_images = True
    config.train_dataset_path = "/data/jkolf/PeriocularExtension/data/PeriocularSyntheticFFHQGan"
    config.synth_type = "synffhqgan"
    config.shuffle      = True
    
    # Settings Backbone
    config.model    = "resnet18" # "resnet18", "resnet50", "mobilefacenet"
    config.emb_size = 512 # 512 fixed
    
    # Settings FP32 Teacher Model
        # Name, will be used to store the model
    config.base_model = None
        # Path to Checkpoint/Weight Dict
    config.base_model_path = None
    
    # Settings Quant. Student Model
        # Folder where model is stored
    config.model_folder : str = "synthmodel"
    config.wq : int = 8
    config.aq : int = 8
    config.quant_batch_size : int = 16
    config.quant_lr : float = 0.01
    config.quant_qat_epochs = 5
    
    # Settings Training
    config.epochs = 10
    config.data_path = "/data/jkolf/PeriocularExtension/data/models"
    config.seed = 1337
    config.log_interval = 50
    config.save_interval= 200
    config.val_interval = 2000
    config.weight_decay = 5e-4
  
    """
        Generate all kind of model pairs
    """
    types_models = ["resnet18", "mobilefacenet"] #"resnet50",
    
    r = 0
  
    for model in types_models:
        for fold in [1,2,3]:
            pretrained_path : str = "/data/jkolf/PeriocularExtension/data/pretrained_models/verification"
            pretrained_path = f"{pretrained_path}/{model}_fp32_verification/backbone_{model}_f{fold}_ver.pth"     
            
            if not Path(pretrained_path).exists():
                raise ValueError("Given pretrained model path does not exist!")
            
            for q_level in [8, 6, 4]:
                
                config.fold  = fold
                config.model = model
                config.base_model = f"{model}_ver_f{fold}"
                config.base_model_path = pretrained_path
                config.model_folder = f"synffhqgan_verification"
                
                config.wq = q_level
                config.aq = q_level
                
                if model == "resnet50":
                    config.quant_batch_size = 8
                else:
                    config.quant_batch_size = 16
                
                with open(f"./synffhqgan_verification/config_ver_synffhqgan_train_{r:05d}.pkl", "wb") as f:
                        pickle.dump(config, f)

                r += 1
