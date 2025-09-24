
import warnings
import torch
from math import isnan
import wandb


# L_p, L_vis, L_o, L_u
class Loss_Tracker:
    def __init__(self, rank):
        self.rank = rank
        
        # Main Losses
        self.losses = {
            "L_p": 0,
            "L_p_rerank": 0,
            "L_vis": 0,
            "L_o": 0,
            "L_u": 0,
            "L_topk_u": 0,
            "L_topk_rank": 0
        }

        self.temporal_count = 0
        self.iteration = 0

    def NaN_check(self, loss, loss_name):
        if isnan(loss):
            warnings.warn(f"Loss {loss_name} is NaN. Check your data and model.")
            return 0
        return loss.item() if isinstance(loss, torch.Tensor) else loss

    def update(self, L_p, L_p_rerank, L_vis, L_o, L_u, L_topk_u, L_topk_rank):
        if self.rank != 0:
            return
        
        L_p = self.NaN_check(L_p, "L_p")
        L_p_rerank = self.NaN_check(L_p_rerank, "L_p_rerank")
        L_vis = self.NaN_check(L_vis, "L_vis")
        L_o = self.NaN_check(L_o, "L_o")
        L_u = self.NaN_check(L_u, "L_u")
        L_topk_u = self.NaN_check(L_topk_u, "L_topk_u")
        L_topk_rank = self.NaN_check(L_topk_rank, "L_topk_rank")

        self.losses["L_p"] += L_p
        self.losses["L_p_rerank"] += L_p_rerank
        self.losses["L_vis"] += L_vis
        self.losses["L_o"] += L_o
        self.losses["L_u"] += L_u
        self.losses["L_topk_u"] += L_topk_u
        self.losses["L_topk_rank"] += L_topk_rank   
        self.temporal_count += 1

    def log(self):
        if self.rank != 0:
            return
        
        L_p = self.losses["L_p"] / self.temporal_count if self.temporal_count > 0 else 0
        L_p_rerank = self.losses["L_p_rerank"] / self.temporal_count if self.temporal_count > 0 else 0
        L_vis = self.losses["L_vis"] / self.temporal_count if self.temporal_count > 0 else 0
        L_o = self.losses["L_o"] / self.temporal_count if self.temporal_count > 0 else 0
        L_u = self.losses["L_u"] / self.temporal_count if self.temporal_count > 0 else 0
        L_topk_u = self.losses["L_topk_u"] / self.temporal_count if self.temporal_count > 0 else 0
        L_topk_rank = self.losses["L_topk_rank"] / self.temporal_count if self.temporal_count > 0 else 0

        # Reset losses and temporal count
        self.losses = {key: 0 for key in self.losses}
        self.temporal_count = 0

        wandb.log({f"Training/point_loss": L_p,
                    f"Training/point_rerank_loss": L_p_rerank,
                    f"Training/visibility_loss": L_vis,
                    f"Training/offset_loss": L_o,
                    f"Training/uncertainty_loss": L_u,
                    f"Training/topk_uncertainty_loss": L_topk_u,
                    f"Training/topk_rank_loss": L_topk_rank,
                    "Training/batch_loss": L_p + L_p + L_vis + L_o + L_u + L_topk_u + L_topk_rank}, commit=False)
        wandb.log({"iteration": self.iteration}, commit=True)

        self.iteration += 1


def init_wandb(args):
    if args.rank == 0:
        project_name = "Track-On2"
        run_name = args.model_save_path.split("/")[-1]
        wandb.init(project=project_name, name=run_name, config=args)

        wandb.define_metric("epoch")
        wandb.define_metric("Evaluation/delta_avg", step_metric="epoch")
        wandb.define_metric("Evaluation/delta_1", step_metric="epoch")
        wandb.define_metric("Evaluation/delta_2", step_metric="epoch")
        wandb.define_metric("Evaluation/delta_4", step_metric="epoch")
        wandb.define_metric("Evaluation/delta_8", step_metric="epoch")
        wandb.define_metric("Evaluation/delta_16", step_metric="epoch")
        wandb.define_metric("Evaluation/AJ", step_metric="epoch")
        wandb.define_metric("Evaluation/OA", step_metric="epoch")

        wandb.define_metric("Evaluation_Train/delta_avg", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/delta_1", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/delta_2", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/delta_4", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/delta_8", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/delta_16", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/AJ", step_metric="epoch")
        wandb.define_metric("Evaluation_Train/OA", step_metric="epoch")

        wandb.define_metric("iteration")
        wandb.define_metric("Training/batch_loss", step_metric="iteration")
        wandb.define_metric("Training/point_loss", step_metric="iteration")
        wandb.define_metric("Training/point_rerank_loss", step_metric="iteration")
        wandb.define_metric("Training/visibility_loss", step_metric="iteration")
        wandb.define_metric("Training/offset_loss", step_metric="iteration")
        wandb.define_metric("Training/uncertainty_loss", step_metric="iteration")
        wandb.define_metric("Training/topk_uncertainty_loss", step_metric="iteration")
        wandb.define_metric("Training/topk_rank_loss", step_metric="iteration")
