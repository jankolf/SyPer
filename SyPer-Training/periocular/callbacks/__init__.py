import torch
import logging
from datetime import datetime
import periocular

from periocular.evaluation.verification import evaluate_verification
from periocular.evaluation.extraction import extract_embeddings

class LoggingCallback:

    def __init__(self, frequency, rank, N_global_steps):

        self.frequency = frequency
        self.rank = rank
        self.N_global_steps = N_global_steps
        self.start_time = datetime.now()

    def start(self):
        self.start_time = datetime.now()

    def __call__(self, current_step, current_loss):
        
        if self.rank != 0:
            return None

        time_delta = (datetime.now() - self.start_time).total_seconds()
        total_runtime = time_delta * (self.N_global_steps - current_step)

        h = int(total_runtime // (60**2))
        m = int((total_runtime % (60**2)) // 60)
        s = int((total_runtime % (60**2)) % 60)

        size_step = str(current_step).rjust(len(str(self.N_global_steps)))

        L = f"Loss={round(current_loss, 3)}".ljust(17)
        S = f"{round(time_delta, 2)}s/Batch".ljust(17)
        R = f"Remaining: {h}h:{m}m:{s}s".ljust(25)

        info = f"[Step {size_step}/{self.N_global_steps}] | {L} | {S} | {R}"
        
        if current_step % self.frequency == 0:
            logging.info(info)

        print(info.ljust(periocular.LINE_WIDTH), end="\r", flush=True)
        
        
class CheckpointCallback:

    def __init__(self, frequency, rank, N_global_steps, model_id, save_path, quantized=False):

        self.frequency = frequency
        self.rank = rank
        self.model_id = model_id
        self.save_path = save_path
        self.N_global_steps = N_global_steps
        self.quantized = quantized

    def __call__(self, current_step, backbone, header, force=False, include_step=True):
        
        if self.rank != 0:
            return None

        if include_step:
            step_info = f"_step{current_step}"
        else:
            step_info = ""

        if force or current_step % self.frequency == 0:

            print(f"[Step {current_step}] Saved models.".ljust(periocular.LINE_WIDTH), end="\n")

            if self.quantized:
                torch.save(backbone.module.state_dict(), f"{self.save_path}/backbone_{self.model_id}{step_info}.pth")
                torch.save(backbone.module, f"{self.save_path}/module_backbone_{self.model_id}{step_info}.pth")
            else:
                torch.save(backbone.module.state_dict(), f"{self.save_path}/backbone_{self.model_id}{step_info}.pth")
                torch.save(backbone.module, f"{self.save_path}/module_backbone_{self.model_id}{step_info}.pth")
                torch.save(header.module.state_dict(), f"{self.save_path}/header_{self.model_id}{step_info}.pth")
                torch.save(header.module, f"{self.save_path}/module_header_{self.model_id}{step_info}.pth")


class VerificationCallback:

    def __init__(self, frequency, rank, N_global_steps, validation_set=None):
        
        self.frequency = frequency
        self.rank = rank
        self.N_global_steps = N_global_steps
        self.validation_set = validation_set

    def __call__(self, current_step, model, force=False):

        if self.rank != 0 or self.validation_set is None:
            return None

        if force or current_step % self.frequency == 0:
            model.eval()
            emb = extract_embeddings(self.validation_set, model, self.rank)
            metrics = evaluate_verification(self.validation_set, emb)
            msg = f"[Eval] [Step {current_step}/{self.N_global_steps}] | AUC={metrics.auc} | EER={metrics.eer} | EER-THR={metrics.eer_thr} | FNMR@1e-3={metrics.fnmr_1e_3} | FNMR@1e-5={metrics.fnmr_1e_5}".ljust(periocular.LINE_WIDTH)
            logging.info(msg)
            print(msg, end="\n")
            model.train()

            return metrics

        return None
