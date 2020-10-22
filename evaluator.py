import json
import logging
import os
from copy import deepcopy
from io import open

import numpy as np
import torch
from tqdm import tqdm

from tools.registry import registry

logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)
logger = logging.getLogger(__name__)


def set_model(model, resume_file):
    logger.info(f"Resuming from Checkpoint: {resume_file}")
    checkpoint = torch.load(resume_file, map_location="cpu")
    new_dict = {}

    for attr in checkpoint["model_state_dict"]:
        if not attr.startswith("module."):
            new_dict["module." + attr] = checkpoint["model_state_dict"][attr]
        else:
            new_dict[attr] = checkpoint["model_state_dict"][attr]

    model.load_state_dict(new_dict)

    # Add checkpoint string
    log_keys = [
        "cs_rank",
        "vqa_rank",
        "vqa_acc",
        "cs_score",
        "global_step",
        "cs_bt_rank",
        "cs_score",
        "cs_bt_score",
    ]
    ckpt_string = f"-------------- \n Checkpoint information: \n"

    for key in log_keys:
        if key in checkpoint:
            ckpt_string += f"{key}: {checkpoint[key]} \n"

    ckpt_string += "---------------"
    logger.info(ckpt_string)
    print("Not loading optimizer and scheduler states")
    del checkpoint

    return model


def final_evaluate(
    evaluate_rephrasings, device, model, dataloaders, save_path, eval_split
):
    if registry["monitor_value"] == "cs_score":
        resume_file = os.path.join(save_path, "cs_best.tar")
    else:
        resume_file = os.path.join(save_path, "vqa_best.tar")

    if not os.path.exists(resume_file):
        import pdb
        pdb.set_trace()
        # raise ValueError("Couldn't find the checkpoint")

    # set model for evaluation
    model = set_model(model, resume_file)

    from mmt.task_utils import forward_eval

    registry.eval_only = True

    model.eval()
    results = {}

    for batch in tqdm(dataloaders[eval_split]):
        with torch.no_grad():  # turn off autograd engine
            batch_dict = forward_eval(device, batch, model, revqa_eval=False)

            # build the json file here!
            logits = torch.max(batch_dict["vil_prediction"], 1)[1].data  # argmax
            for idx in range(logits.size(0)):
                results[batch_dict["question_id"][idx].item()] = {
                    "question_id": batch_dict["question_id"][idx].item(),
                    "answer": registry.label2ans[logits[idx].item()],
                    "vqa_score": np.round(batch_dict["vqa_scores"][idx], 1)
                    if "vqa_scores" in batch_dict
                    else None,
                }

    human_cs_scores, bt_cs_scores = None, None
    if registry.revqa_eval and eval_split == "val":
        human_cs_scores, bt_cs_scores = evaluate_rephrasings(dataloaders, model, device)

    final_results = {}
    final_results["results"] = results
    final_results["human_cs_scores"] = human_cs_scores
    final_results["bt_cs_scores"] = bt_cs_scores

    evalai_results = deepcopy(list(results.values()))
    for item in evalai_results:
        del item["vqa_score"]

    save_dir = os.path.split(resume_file)[0]
    evalai_path = f"{save_dir}/evalai_{eval_split}.json"
    preds_path = f"{save_dir}/preds_revqa_{eval_split}.json"

    # dump eval-ai file and results file
    json.dump(evalai_results, open(evalai_path, "w"))
    json.dump(final_results, open(preds_path, "w"))

    print(f"Dumped: {evalai_path}")
    print(f"Dumped: {preds_path}")

    model.train()
