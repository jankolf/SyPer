from types import SimpleNamespace
import pickle
from pathlib import Path

if __name__ == "__main__":

    config = SimpleNamespace()

    # Settings Train Dataset
    config.train_dataset_type = "UFPR"
    config.fold = 1 # 1,2,3
    config.protocol = "closed_world"
    config.img_size     = 224
    config.flip_images = True
    config.synth_type = "pretrainedfp32"
    config.shuffle      = True
    config.emb_size = 512
    
    # Settings Quant. Student Model
        # Folder where model is stored
    config.model_folder = "synthmodel"
    config.wq = 8
    config.aq = 8

    # Settings Training
    config.epochs = 10
    config.data_path = "/data/jkolf/PeriocularExtension/data/pretrained_models"
    config.results_path = "/data/jkolf/PeriocularExtension/data/results"
    config.model_folder = "pretrainedfp32"
    config.seed = 1337
  
    """
        Generate all kind of model pairs
    """
    types_models = ["resnet18", "resnet50", "mobilefacenet"]
    
    r : int = 0
    
    flipping : str = "_flip" if config.flip_images else ""
    
    for scenario, protocol in [("identification","closed_world"), ("verification","open_world_valclosed")]:
        
        config.protocol = protocol
        
        if scenario == "identification":
            label = "id"
        else:
            label = "ver"
        
        for model in types_models:
            for fold in [1,2,3]:
                    
                config.fold  = fold
                config.model = model
                config.base_model = f""
                
                config.wq = "None"
                config.aq = "None"
                
                config.results_file = f"{config.results_path}/{model}_{scenario}_pretrainedfp32_nirxspectrum.txt"
                
                config.test_model_id = f"{model}_{scenario}_f{fold}_fp32"
                config.test_module_path = f"{config.data_path}/{scenario}/{model}_fp32_{scenario}/backbone_{model}_f{fold}_{label}.pth"
                
                if not Path(config.test_module_path).exists():
                    print(f"Model {config.test_model_id} not found under path {config.test_module_path}!")
                    continue
                
                with open(f"./pretrained/config_fp32_nirxspectrum_test_{r:05d}.pkl", "wb") as f:
                        pickle.dump(config, f)

                r += 1
