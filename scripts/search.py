import argparse
import datetime
import os
import random
import shutil
import sys
from pprint import pprint

import mlflow
import numpy as np
import optuna
import paddle

import paddleseg.transforms as T
from paddleseg.datasets import Dataset
from paddleseg.models import PPLiteSeg
from paddleseg.models.backbones.stdcnet import STDC2
from paddleseg.models.losses import CrossEntropyLoss, DiceLoss

# Setup device and seed
paddle.set_device('gpu')
SEED = 42
paddle.seed(SEED)
np.random.seed(SEED)
random.seed(SEED)

# Added global hyperparameters
BATCH_SIZE = 6
MAX_ITERATIONS = 5000
EVAL_INTERVAL = 100
PATIENCE = 10

# MLflow experiment setup
mlflow.set_experiment("PPLiteSeg_OilSpill")

# Dataset paths and transformations
DATASET_ROOT = "data/train"
TRAIN_LIST = "data/train/train.txt"
VAL_LIST   = "data/train/val.txt"

transforms = [
    T.Resize(target_size=(1024, 1024)),
    T.Normalize()
]

train_dataset = Dataset(dataset_root=DATASET_ROOT, train_path=TRAIN_LIST,
                        transforms=transforms, img_channels=2, num_classes=2, mode='train')
val_dataset = Dataset(dataset_root=DATASET_ROOT, val_path=VAL_LIST,
                      transforms=transforms, img_channels=2, num_classes=2, mode='val')

# Helper function for a single training batch
def train_one_batch(model, data, loss_fn, optimizer):
    images = data['img']
    labels = data['label'].astype('int64')
    outputs = model(images)
    if isinstance(outputs, (list, tuple)):
        loss_value = sum([loss_fn(logit, labels) for logit in outputs])
    else:
        loss_value = loss_fn(outputs, labels)
    loss_value.backward()
    optimizer.step()
    optimizer.clear_grad()
    return float(loss_value)

# Helper function to evaluate model performance
def evaluate_model(model, val_loader):
    print("Starting evaluation")
    total_intersect_c = [0, 0]
    total_union_c = [0, 0]
    total_tp_c = [0, 0]
    total_fp_c = [0, 0]
    total_fn_c = [0, 0]
    with paddle.no_grad():
        for data in val_loader:
            images = data['img']
            labels = data['label'].astype('int64')
            trans_info = data.get('trans_info', None)
            outputs = model(images)
            if isinstance(outputs, (list, tuple)):
                outputs = outputs[0]
            if trans_info is not None:
                from paddleseg.core.infer import reverse_transform
                outputs = reverse_transform(outputs, trans_info, mode='bilinear')
            preds = outputs.argmax(axis=1).numpy()
            label_mask = labels.numpy()
            for c in [0, 1]:
                tp = np.logical_and(preds == c, label_mask == c).sum()
                total_intersect_c[c] += tp
                total_union_c[c] += np.logical_or(preds == c, label_mask == c).sum()
                total_tp_c[c] += tp
                total_fp_c[c] += (preds == c).sum() - tp
                total_fn_c[c] += (label_mask == c).sum() - tp
    val_iou_class0 = total_intersect_c[0] / total_union_c[0] if total_union_c[0] > 0 else 0.0
    val_iou_class1 = total_intersect_c[1] / total_union_c[1] if total_union_c[1] > 0 else 0.0
    val_iou = (val_iou_class0 + val_iou_class1) / 2
    val_precision_class0 = total_tp_c[0] / (total_tp_c[0] + total_fp_c[0]) if (total_tp_c[0] + total_fp_c[0]) > 0 else 0.0
    val_precision_class1 = total_tp_c[1] / (total_tp_c[1] + total_fp_c[1]) if (total_tp_c[1] + total_fp_c[1]) > 0 else 0.0
    val_recall_class0 = total_tp_c[0] / (total_tp_c[0] + total_fn_c[0]) if (total_tp_c[0] + total_fn_c[0]) > 0 else 0.0
    val_recall_class1 = total_tp_c[1] / (total_tp_c[1] + total_fn_c[1]) if (total_tp_c[1] + total_fn_c[1]) > 0 else 0.0
    return {"val_iou_class0": val_iou_class0, "val_iou_class1": val_iou_class1,
            "val_iou": val_iou,
            "val_precision_class0": val_precision_class0, "val_precision_class1": val_precision_class1,
            "val_recall_class0": val_recall_class0, "val_recall_class1": val_recall_class1}

# Objective function for hyperparameter tuning with Optuna
def objective(trial):
    print(f"Starting trial {trial.number}")
    loss_name    = trial.suggest_categorical("loss_function", ["CrossEntropy", "Dice"])
    optim_name   = trial.suggest_categorical("optimizer", ["Adam", "SGD"])
    lr_schedule  = trial.suggest_categorical("lr_schedule", ["poly", "poly_warmup"])
    base_lr      = trial.suggest_float("learning_rate", 1e-5, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-6, 1e-2, log=True)
    warmup_steps = trial.suggest_int("warmup_steps", 100, 1000, step=100) if lr_schedule == "poly_warmup" else None

    # Load pretrained model
    pretrain_path = "https://bj.bcebos.com/paddleseg/dygraph/PP_STDCNet2.tar.gz"
    model = PPLiteSeg(num_classes=2,
                      backbone=STDC2(in_channels=2, pretrained=pretrain_path),
                      pretrained=None)
    
    loss_fn = CrossEntropyLoss() if loss_name == "CrossEntropy" else DiceLoss()
    
    if lr_schedule == "poly":
        lr_scheduler = paddle.optimizer.lr.PolynomialDecay(learning_rate=base_lr, decay_steps=MAX_ITERATIONS, end_lr=0, power=0.9)
    else:
        base_sched = paddle.optimizer.lr.PolynomialDecay(learning_rate=base_lr, decay_steps=MAX_ITERATIONS, end_lr=0, power=0.9)
        lr_scheduler = paddle.optimizer.lr.LinearWarmup(learning_rate=base_sched, warmup_steps=warmup_steps, start_lr=0.0, end_lr=base_lr)
    
    optimizer = (paddle.optimizer.Adam(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=weight_decay)
                 if optim_name == "Adam" else
                 paddle.optimizer.SGD(learning_rate=lr_scheduler, parameters=model.parameters(), weight_decay=weight_decay))

    # Data loaders
    train_loader = paddle.io.DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, drop_last=True)
    val_loader = paddle.io.DataLoader(val_dataset, batch_size=1, shuffle=False)

    best_iou = 0.0
    patience_counter = 0
    total_loss = 0.0
    iteration = 1
    epoch = 1

    with mlflow.start_run(run_name=f"trial_{trial.number}", nested=True):
        mlflow.log_params({
            "loss_function": loss_name,
            "optimizer": optim_name,
            "lr_schedule": lr_schedule,
            "learning_rate": base_lr,
            "weight_decay": weight_decay,
            "warmup_steps": warmup_steps
        })
        print(f"Trial {trial.number} parameters:")
        pprint(trial.params)

        early_stop = False
        while iteration <= MAX_ITERATIONS and not early_stop:
            print(f"Starting epoch {epoch}")
            mlflow.log_metric("epoch", epoch, step=iteration)
            for data in train_loader:
                if iteration > MAX_ITERATIONS:
                    break
                model.train()
                batch_loss = train_one_batch(model, data, loss_fn, optimizer)
                if isinstance(lr_scheduler, paddle.optimizer.lr.LRScheduler):
                    if isinstance(lr_scheduler, paddle.optimizer.lr.ReduceOnPlateau):
                        lr_scheduler.step(batch_loss)
                    else:
                        lr_scheduler.step()
                total_loss += batch_loss

                if iteration % 10 == 0 or iteration == 1:
                    avg_loss = total_loss / 10
                    mlflow.log_metrics({"train_loss": avg_loss,
                                        "learning_rate": lr_scheduler.get_lr()}, step=iteration)
                    print(f"Iteration {iteration}: Avg Loss = {avg_loss:.4f}, LR = {lr_scheduler.get_lr():.6f}, Epoch = {epoch}")
                    total_loss = 0.0

                if iteration % EVAL_INTERVAL == 0:
                    eval_metrics = evaluate_model(model, val_loader)
                    mlflow.log_metrics(eval_metrics, step=iteration)
                    print(f"----- Evaluation at iteration {iteration} -----")
                    print(f"Mean IoU: {eval_metrics['val_iou']:.4f}")
                    print(f"IoU: [{eval_metrics['val_iou_class0']:.4f} {eval_metrics['val_iou_class1']:.4f}]")
                    print(f"Precision: [{eval_metrics['val_precision_class0']:.4f} {eval_metrics['val_precision_class1']:.4f}]")
                    print(f"Recall: [{eval_metrics['val_recall_class0']:.4f} {eval_metrics['val_recall_class1']:.4f}]")
                    
                    base_dir = f"output/optuna/trial_{trial.number}"
                    os.makedirs(base_dir, exist_ok=True)
                    iter_dir = f"{base_dir}/iter_{iteration}"
                    os.makedirs(iter_dir, exist_ok=True)
                    paddle.save(model.state_dict(), f"{iter_dir}/model.pdparams")
                    
                    # Rotation: keep only the 5 most recent iteration directories
                    iter_dirs = [d for d in os.listdir(base_dir) if d.startswith("iter_")]
                    if len(iter_dirs) > 5:
                        iter_dirs_sorted = sorted(iter_dirs, key=lambda x: int(x.split('_')[1]))
                        for old_dir in iter_dirs_sorted[:-5]:
                            full_old_path = os.path.join(base_dir, old_dir)
                            shutil.rmtree(full_old_path)
                    
                    if eval_metrics["val_iou"] > best_iou:
                        best_iou = eval_metrics["val_iou"]
                        patience_counter = 0
                        best_dir = f"{base_dir}/best_model"
                        os.makedirs(best_dir, exist_ok=True)
                        paddle.save(model.state_dict(), f"{best_dir}/model.pdparams")
                        print(f"New best model saved at {best_dir} with IoU: {best_iou:.4f} at iteration {iteration}")
                    else:
                        patience_counter += 1
                        print(f"No improvement at iteration {iteration}: patience {patience_counter}/{PATIENCE}")
                        if patience_counter >= PATIENCE:
                            print("Patience threshold reached, stopping trial.")
                            early_stop = True
                            break
                iteration += 1
            epoch += 1

        mlflow.log_metric("best_val_iou", best_iou)
    print(f"Trial {trial.number} complete with best IoU: {best_iou:.4f}")
    return best_iou

# Main entry point
def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--resume", type=str, default=None)
    args = parser.parse_args()
    db_path = "sqlite:///output/optuna_study.db"

    if args.resume:
        print(f"Resuming run with name: {args.resume}")
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name("PPLiteSeg_OilSpill")
        runs = client.search_runs([experiment.experiment_id], filter_string=f"tags.mlflow.runName = '{args.resume}'")
        if not runs:
            print(f"No run found with name: {args.resume}")
            sys.exit(1)
        resumed_run = runs[0]
        with mlflow.start_run(run_id=resumed_run.info.run_id, nested=True):
            # Use the provided run name for loading the study
            study = optuna.load_study(study_name=args.resume, storage=db_path)
            # Reschedule failed trials
            for t in study.trials:
                if t.state == optuna.trial.TrialState.FAIL:
                    study.enqueue_trial(t.params)
            study.optimize(objective, n_trials=10)
    else:
        print("Initiating a new run")
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        run_name = f"Optuna_Study_{timestamp}"
        with mlflow.start_run(run_name=run_name):
            mlflow.log_params({
                "batch_size": BATCH_SIZE,
                "max_iterations": MAX_ITERATIONS,
                "eval_interval": EVAL_INTERVAL,
                "patience": PATIENCE,
                "paddleseg_model": "PPLiteSeg",
                "backbone": "STDC2"
            })
            study = optuna.create_study(direction="maximize", storage=db_path, study_name=run_name, load_if_exists=False)
            fixed_trial = {
                "loss_function": "CrossEntropy",
                "optimizer": "Adam",
                "lr_schedule": "poly",
                "learning_rate": 0.01,
                "weight_decay": 4.0e-5,
            }
            study.enqueue_trial(fixed_trial)
            study.optimize(objective, n_trials=10)

    best_trial = study.best_trial
    print(f"Best IoU: {best_trial.value:.4f}")
    print("Best hyperparameters:", best_trial.params)

if __name__ == '__main__':
    main()
