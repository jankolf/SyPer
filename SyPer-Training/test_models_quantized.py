from pathlib import Path
import os
import sys
import traceback
from tqdm import tqdm
import pickle


import torch

from periocular.backbones import get_model
from periocular.datasets import PeriocularTest
from periocular.datasets import ProtocolType
from periocular.evaluation.extraction import extract_embeddings
from periocular.evaluation.ranking import evaluate_rank_accuracy
from periocular.evaluation.verification import evaluate_verification

def get_last_checkpoint(model_directory):
    
    
    best_backbone = model_directory / f"module_backbone_{model_directory.name}.pth"
    
    if best_backbone.exists():
        return str(best_backbone)

    sel = max([int(str(d).split("step")[-1].split(".")[0]) 
                 for d in model_directory.glob("module_backbone_*") if "_step" in str(d)])
    model_path = list(model_directory.glob(f"module_backbone_*{sel}*"))[0]
    
    return model_path
    

def build_string(metrics_verification, metrics_ranking):
    r1 : str = f"Rank1={round(metrics_ranking.rank_1_acc, 6)}"
    r5 : str = f"Rank5={round(metrics_ranking.rank_5_acc, 6)}"
    auc : str = f"AUC={round(metrics_verification.auc,6)}"
    eer : str = f"EER={round(metrics_verification.eer,6)}"
    fnmr1e2 : str = f"FNMR@FMR1e-2={metrics_verification.fnmr_1e_2}" 
    fnmr1e3 : str = f"FNMR@FMR1e-3={metrics_verification.fnmr_1e_3}" 
    fnmr1e4 : str = f"FNMR@FMR1e-4={metrics_verification.fnmr_1e_4}" 
    fnmr1e5 : str = f"FNMR@FMR1e-5={metrics_verification.fnmr_1e_5}" 
    fnmr1e6 : str = f"FNMR@FMR1e-6={metrics_verification.fnmr_1e_6}"
    
    return f"{r1};{r5};{auc};{eer};{fnmr1e2};{fnmr1e3};{fnmr1e4};{fnmr1e5};{fnmr1e6}"

if __name__ == "__main__":

    config_file = [f for f in sys.argv if "pkl" in f][0]
    
    with open(f"{config_file}", "rb") as f:
        config = pickle.load(f)

    print(f"Testing Model {config.test_model_id}")

    protocol = {
                "open_world_valopen":ProtocolType.OPEN_WORLD_OPEN_VAL, 
                "closed_world":ProtocolType.CLOSED_WORLD,
                "open_world_valclosed":ProtocolType.OPEN_WORLD_CLOSED_VAL
            }[config.protocol]

    FLIP_L2R = config.flip_images
    
    line_info = f"[{config.test_model_id}][{config.protocol}][f{config.fold}][q{config.wq}]&>"

    result_path = config.results_file


    test_set = PeriocularTest(protocol, config.fold, flip_L_to_R=FLIP_L2R)

    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    try:
        backbone = torch.load(config.test_module_path)
        model = torch.nn.DataParallel(backbone)
        model.eval()

        embeddings = extract_embeddings(test_set, model, device)
        metrics_verification = evaluate_verification(test_set, embeddings)
        metrics_ranking = evaluate_rank_accuracy(test_set, embeddings)

        res = build_string(metrics_verification, metrics_ranking)
        line = f"{line_info}{res}\n"
        
        with open(result_path, "a") as f:
            f.write(line)
            
    except KeyboardInterrupt:
        sys.exit(0)
    except Exception as e:
        traceback.print_exc()