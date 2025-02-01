
import warnings
import torch
from math import isnan
import wandb


loss_acc = {}
def log_batch_loss(args, optimizer, dataloader, total_update_num, i, out):
    if args.rank == 0:
        global loss_acc

        
        for key, value in out.items():
            if "loss" in key:
                loss_acc[key] = loss_acc.get(key, 0) + value.item()
        lr = optimizer.state_dict()["param_groups"][0]["lr"]

        total_loss = 0
        for key, value in loss_acc.items():

            wandb.log({f"Training/{key}": value}, commit=False)

            total_loss += value

        dataloader.set_description(f"lr: {lr:.9f} | loss: {total_loss:.5f}")
        wandb.log({"Training/batch_loss": total_loss, "iteration": total_update_num}, commit=True)

        # reset loss accumulator
        loss_acc = {}

def log_epoch_loss(args, total_loss, epoch, dataloader):
    if args.rank == 0:
        mean_loss = total_loss / (len(dataloader))
        wandb.log({"Training/epoch_loss": mean_loss}, commit=False)
        print(f"Mean loss: {mean_loss:.3f}")


def log_eval_metrics(args, results, epoch):
    if args.rank == 0:

        if not args.validation and not args.online_validation:
            wandb.log({"Evaluation/delta_avg": results["delta_avg"],
                        "Evaluation/delta_1": results["delta_1"],
                        "Evaluation/delta_2": results["delta_2"],
                        "Evaluation/delta_4": results["delta_4"],
                        "Evaluation/delta_8": results["delta_8"],
                        "Evaluation/delta_16": results["delta_16"],
                        "Evaluation/AJ": results["aj"],
                        "Evaluation/OA": results["oa"],
                        "epoch": epoch}, commit=True)

        print(f"delta_avg: {results['delta_avg']:.2f}")
        print(f"delta_1: {results['delta_1']:.2f}")
        print(f"delta_2: {results['delta_2']:.2f}")
        print(f"delta_4: {results['delta_4']:.2f}")
        print(f"delta_8: {results['delta_8']:.2f}")
        print(f"delta_16: {results['delta_16']:.2f}")
        print(f"AJ: {results['aj']:.2f}")
        print(f"OA: {results['oa']:.2f}")

def init_wandb(args):
    if args.rank == 0:
        project_name = "Track-On"
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

        wandb.define_metric("Training/epoch_loss", step_metric="epoch")

        wandb.define_metric("iteration")
        wandb.define_metric("Training/batch_loss", step_metric="iteration")
        wandb.define_metric("Training/point_loss", step_metric="iteration")
        wandb.define_metric("Training/visibility_loss", step_metric="iteration")
        wandb.define_metric("Training/offset_loss", step_metric="iteration")
