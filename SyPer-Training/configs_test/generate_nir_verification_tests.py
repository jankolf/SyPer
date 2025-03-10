from types import SimpleNamespace
import pickle
from pathlib import Path

if __name__ == "__main__":

    config = SimpleNamespace()

    # Settings Train Dataset
    config.train_dataset_type = "NIR"
    config.fold = 1 # 1,2,3
    config.protocol = "Testing"
    config.img_size     = 224
    config.flip_images = True
    config.synth_type = "NIR-Images"
    config.shuffle      = True
    
    # Settings Quant. Student Model
        # Folder where model is stored
    config.model_folder : str = "nirmodel"
    config.wq = 8
    config.aq = 8

    # Settings Training
    config.epochs = 10
    config.data_path = "/data/jkolf/PeriocularExtension/data/models"
    config.results_path = "/data/jkolf/PeriocularExtension/data/results"
    config.model_folder = "nir_verification"
    config.seed = 1337
  
    """
        Generate all kind of model pairs
    """
    types_models = ["resnet18", "resnet50", "mobilefacenet"]
    
    r : int = 0
    
    flipping : str = "_flip" if config.flip_images else ""
      
    for model in types_models:
        for fold in [1,2,3]:
            for q_level in [8, 6, 4]:
                
                steps = [1575, 1800, 2025, 2250]
                if model == "resnet50":
                    steps = [3150, 3600, 4050, 4500]
                
                for step in steps:
                    config.fold  = fold
                    config.model = model
                    config.base_model = f""
                    
                    config.wq = q_level
                    config.aq = q_level
                    
                    config.results_file = f"{config.results_path}/{model}_ver_q{q_level}_nir{flipping}"
                    
                    config.test_model_id = f"{model}_ver_f{fold}_w{q_level}a{q_level}_NIR-Images{flipping}"
                    config.test_module_path = f"{config.data_path}/{config.model_folder}/{config.test_model_id}/module_backbone_{config.test_model_id}_step{step}.pth"
                    
                    config.results_file = f"{config.results_path}/{model}_ver_f-all_w{q_level}a{q_level}_nir{flipping}.txt"
                    
                    if not Path(config.test_module_path).exists():
                        print(f"Model {config.test_model_id} not found under path {config.test_module_path}!")
                        continue
                    
                    with open(f"./nir_verification/config_ver_nir_test_{r:05d}.pkl", "wb") as f:
                            pickle.dump(config, f)

                    r += 1
