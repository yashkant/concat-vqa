import argparse
import json
import logging
import os
import random
from builtins import ValueError
from collections import defaultdict
from io import open

import numpy as np
import torch
import yaml
from easydict import EasyDict as edict
from tqdm import tqdm

from evaluator import final_evaluate
from mmt.metrics import get_consistency_score
from tools.registry import registry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


logger = logging.getLogger(__name__)


def get_config():
    """ Set default and command line arguments. """
    parser = argparse.ArgumentParser()
    parser.add_argument("--task_file", required=True, type=str, help="joint config file")
    parser.add_argument("--tag", required=True, type=str, help="tag for the experiment")

    args = parser.parse_args()
    with open(args.task_file, "r") as f:
        task_cfg = edict(yaml.safe_load(f))

    set_seeds(task_cfg)
    registry.update(task_cfg)

    logger.info("-" * 20 + "Config Start" + "-" * 20)
    print(json.dumps(vars(args), indent=2))
    print(json.dumps(vars(task_cfg), indent=2))
    logger.info("-" * 20 + "Config End" + "-" * 20)

    return task_cfg, args


def set_seeds(task_cfg):
    """ Set seeds for reproducibility """
    seed = task_cfg["seed"]
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


def set_device_folder(task_cfg, args):
    if torch.cuda.is_available():
        device = torch.device("cuda")
        multi_gpu = torch.cuda.device_count() > 1
    else:
        raise ValueError("Cuda not available!")

    # build experiment directory
    save_path = os.path.join("save", args.tag)
    if not os.path.exists(save_path):
        os.makedirs(save_path)

    # dump full experiment configuration (helps in reproducibility)
    with open(os.path.join(save_path, "command.txt"), "w") as f:
        print(args, file=f)  # Python 3.x
        print("\n", file=f)
        print(task_cfg, file=f)

    return device, multi_gpu, save_path


def build_checkpoint(
    model, optimizer, warmup_scheduler, global_step, vqa_score, cs_scores, cs_bt_scores
):
    """ Generate a storable checkpoint from model"""
    model_to_save = model.module if hasattr(model, "module") else model
    checkpoint_dict = {
        "model_state_dict": model_to_save.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "warmup_scheduler_state_dict": warmup_scheduler.state_dict(),
        "global_step": global_step,
        "vqa_score": vqa_score,
        "cs_scores": cs_scores,
        "cs_bt_scores": cs_bt_scores,
    }
    return checkpoint_dict


def main():
    """ Trains a model and evaluates it."""
    task_cfg, args = get_config()

    from mmt.mmt import MMT, BertConfig
    from mmt.task_utils import (
        clip_gradients,
        forward_train,
        get_optim_scheduler,
        load_dataset,
    )

    base_lr = task_cfg["lr"]

    device, multi_gpu, save_path = set_device_folder(task_cfg, args)

    # load datasets
    dataloaders = load_dataset(task_cfg)

    # build model
    mmt_config = BertConfig.from_dict(task_cfg["MMT"])
    text_bert_config = BertConfig.from_dict(task_cfg["TextBERT"])
    model = MMT(mmt_config, text_bert_config)

    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    logger.info(f"Training Parameters: {trainable_params}")

    # load optimizers
    optimizer_grouped_parameters = model.get_optimizer_parameters(base_lr)
    optimizer, warmup_scheduler = get_optim_scheduler(
        task_cfg, optimizer_grouped_parameters, base_lr
    )

    # send to gpu
    model.to(device)
    for state in optimizer.state.values():
        for k, v in state.items():
            if torch.is_tensor(v):
                state[k] = v.cuda()
    if multi_gpu:
        model = torch.nn.DataParallel(model)

    # store best values
    eval_iter_factor = task_cfg["eval_iter_factor"]
    best_vqa, best_cs = -1, -1
    loss_hist, score_hist = [], []
    global_step = 0
    start_epoch = 0
    eval_ckpts_file = os.path.join(save_path, "ckpts.txt")

    # train loop
    num_iters = len(dataloaders["train_scl"] if registry.alt_train else dataloaders["train_ce"])
    model.train()

    for epochId in tqdm(range(start_epoch, task_cfg["num_epoch"]), desc="Epoch"):
        for step in tqdm(range(num_iters), desc="Iters"):

            assert model.training
            if global_step > registry.hard_stop:
                logger.info(f"Breaking w/ hard-stop at {registry.hard_stop}")
                break

            iter_id = step + (epochId * num_iters)

            # set run-type ("scl" vs "ce")
            if registry.alt_train and iter_id % registry.ce_freq == 1:
                train_type = "scl"
            else:
                train_type = "ce"

            loss, score = forward_train(device, dataloaders, model, train_type)

            loss.backward()

            if task_cfg["grad_clip_mode"] == "all":
                clip_gradients(
                    model, task_cfg["max_grad_norm"], task_cfg["grad_clip_mode"]
                )

            optimizer.step()
            warmup_scheduler.step()
            model.zero_grad()
            optimizer.zero_grad()
            global_step += 1

            if train_type == "ce" or (not registry.alt_train):
                loss_hist.append(float(loss))
                score_hist.append(float(score))

            del loss
            del score

            if step % 20 == 0 and step != 0:
                logger.info(
                    f"Score: {sum(score_hist)/len(score_hist)}, Loss: {sum(loss_hist)/len(loss_hist)}"
                )
                loss_hist, score_hist = [], []

            if (iter_id != 0 and iter_id % eval_iter_factor == 0) or (
                global_step == registry.hard_stop
            ):
                logger.info("Starting Validation Run....")
                curr_val_score, curr_val_loss, cs_scores, cs_bt_scores = run_evaluation(
                    dataloaders, device, model
                )

                # log current results
                ckpt_string = f"Iter: {global_step} | VQA: {curr_val_score} | CS: {cs_scores} | CS-BT: {cs_bt_scores}"
                with open(eval_ckpts_file, "a") as f:
                    f.write(ckpt_string + "\n")
                logger.info(ckpt_string)

                # build dict for storing the checkpoint
                checkpoint_dict = build_checkpoint(
                    model,
                    optimizer,
                    warmup_scheduler,
                    global_step,
                    curr_val_score,
                    cs_scores,
                    cs_bt_scores,
                )

                # checkpoint based on best vqa-score
                if task_cfg["monitor_value"] == "vqa_score":
                    if best_vqa < curr_val_score:
                        output_checkpoint = os.path.join(save_path, f"vqa_best.tar")
                        torch.save(checkpoint_dict, output_checkpoint)
                        best_vqa = curr_val_score
                    logger.info(f"Monitoring vqa-score, best: {best_vqa} | current: {curr_val_score}")

                # checkpoint based on best cs-score on back-translation rephrasings
                elif task_cfg["monitor_value"] == "cs_score":
                    if best_cs < cs_bt_scores[-1]:
                        output_checkpoint = os.path.join(save_path, f"cs_best.tar")
                        torch.save(checkpoint_dict, output_checkpoint)
                        best_cs = cs_bt_scores[-1]
                        logger.info(f"Monitoring CS-4 score, best: {best_cs} | current: {cs_bt_scores[-1]}")
                else:
                    raise ValueError


        # break at hard-stop
        if global_step > registry.hard_stop:
            break

    # Run final-evaluation and generate the EvalAI files.
    for split in ["test", "val"]:
        final_evaluate(
            evaluate_rephrasings, device, model, dataloaders, save_path, split
        )


def reset_evaluation_bins():
    """ Reset rephrasing bins for each evaluation """
    if registry.revqa_eval:
        from easydict import EasyDict

        dd = defaultdict(list)
        dd_bt = defaultdict(list)

        super(EasyDict, registry).__setattr__("revqa_bins", dd)
        super(EasyDict, registry).__setitem__("revqa_bins", dd)

        super(EasyDict, registry).__setattr__("revqa_bt_bins", dd_bt)
        super(EasyDict, registry).__setitem__("revqa_bt_bins", dd_bt)


def evaluate_rephrasings(dataloaders, model, device):
    """ Run evaluation on human and back-translated rephrasings """
    from mmt.task_utils import forward_eval

    reset_evaluation_bins()
    for batch in tqdm(dataloaders["revqa"], desc="Evaluate (Human Rephrasings)"):
        with torch.no_grad():  # turn off autograd engine
            forward_eval(device, batch, model, revqa_eval=True, revqa_split="revqa")
    # collect consensus results
    human_cs_scores = get_consistency_score(bins_key="revqa_bins")

    for batch in tqdm(
        dataloaders["revqa_bt"], desc="Evaluate (Back Translated Rephrasings)"
    ):
        with torch.no_grad():  # turn off autograd engine
            forward_eval(device, batch, model, revqa_eval=True, revqa_split="revqa_bt")

    # collect consensus results
    bt_cs_scores = get_consistency_score(bins_key="revqa_bt_bins")

    # filter out consensus scores
    bt_cs_scores = [bt_cs_scores[key] for key in ["1_bt", "2_bt", "3_bt", "4_bt"]]
    human_cs_scores = [human_cs_scores[str(key)] for key in [1, 2, 3, 4]]

    return human_cs_scores, bt_cs_scores


def run_evaluation(
    dataloaders,
    device,
    model,
):
    """ Run evaluation on minival (VQA-score) and rephrasings (Consensus Scores) """
    from mmt.task_utils import forward_eval

    model.eval()  # turn off dropout/batch-norm

    # run on validation-set
    val_scores, val_losses, batch_sizes = [], [], []
    for i, batch in tqdm(
        enumerate(dataloaders["minval"]),
        total=len(dataloaders["minval"]),
        desc="Evaluate (Mini-Val)",
    ):
        with torch.no_grad():  # turn off autograd engine
            loss, score, batch_size = forward_eval(
                device, batch, model, revqa_eval=False
            )
            val_scores.append(score * batch_size)
            val_losses.append(loss * batch_size)
            batch_sizes.append(batch_size)

    # run consensus evaluation on human and back-translated rephrasings
    if registry.revqa_eval:
        human_cs_scores, bt_cs_scores = evaluate_rephrasings(dataloaders, model, device)
    else:
        human_cs_scores, bt_cs_scores = None, None

    vqa_score = sum(val_scores) / sum(batch_sizes)
    vqa_loss = sum(val_losses) / sum(batch_sizes)

    model.train()  # return to train state

    return vqa_score, vqa_loss, human_cs_scores, bt_cs_scores


if __name__ == "__main__":
    main()
