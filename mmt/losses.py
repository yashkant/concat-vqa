import logging

import torch

from tools.registry import registry

logger = logging.getLogger(__name__)


class ScaledSupConLoss(torch.nn.Module):
    """Scaled Supervised Contrastive Learning: https://arxiv.org/pdf/2004.11362.pdf.
    It also supports the unsupervised contrastive loss in SimCLR"""

    def __init__(
        self,
        temperature=0.07,
        contrast_mode="all",
        base_temperature=0.07,
        formulation="normal",
    ):
        super(ScaledSupConLoss, self).__init__()
        self.temperature = temperature
        self.contrast_mode = contrast_mode
        self.base_temperature = base_temperature
        self.formulation = formulation
        logger.info(
            f"SCL Loss Formulation: {self.formulation} w/ BT: {base_temperature} and T: {temperature}"
        )

    def set_formulation(self, formulation):
        self.formulation = formulation

    def forward(self, batch_dict):
        """Compute loss for model. If both `labels` and `mask` are None,
        it degenerates to SimCLR unsupervised loss:
        https://arxiv.org/pdf/2002.05709.pdf

        Args:
            features: hidden vector of shape [bsz, n_views, ...].
            labels: ground truth of shape [bsz].
            mask: contrastive mask of shape [bsz, bsz], mask_{i,j}=1 if sample j
                has the same class as sample i. Can be asymmetric.
        Returns:
            A loss scalar.
        """

        # join features
        features = torch.cat(
            [bd["contrastive_projection_norm"].unsqueeze(dim=1) for bd in batch_dict],
            dim=1,
        )

        # targets for the batch is the one with highest score
        labels = batch_dict[0]["target"].argmax(dim=-1).view(-1, 1)

        # samples without an answer cannot work as anchor points
        mask_samples = (batch_dict[0]["target"].sum(dim=-1) != 0).int()

        # mask
        pos_mask = None

        device = torch.device("cuda") if features.is_cuda else torch.device("cpu")

        if len(features.shape) < 3:
            raise ValueError(
                "`features` needs to be [bsz, n_views, ...],"
                "at least 3 dimensions are required"
            )
        if len(features.shape) > 3:
            features = features.view(features.shape[0], features.shape[1], -1)

        batch_size = features.shape[0]
        if labels is not None and pos_mask is not None:
            raise ValueError("Cannot define both `labels` and `mask`")
        elif labels is None and pos_mask is None:
            pos_mask = torch.eye(batch_size, dtype=torch.float32).to(device)
        elif labels is not None:
            labels = labels.contiguous().view(-1, 1)
            if labels.shape[0] != batch_size:
                raise ValueError("Num of labels does not match num of features")
            pos_mask = torch.eq(labels, labels.T).float().to(device)
        else:
            pos_mask = pos_mask.float().to(device)

        # remove samples without gt
        pos_mask = pos_mask * mask_samples
        contrast_count = features.shape[1]

        contrast_feature = torch.cat(torch.unbind(features, dim=1), dim=0)
        if self.contrast_mode == "one":
            anchor_feature = features[:, 0]
            anchor_count = 1
        elif self.contrast_mode == "all":
            anchor_feature = contrast_feature
            anchor_count = contrast_count
        else:
            raise ValueError("Unknown mode: {}".format(self.contrast_mode))

        # compute logits
        anchor_dot_contrast = torch.div(
            torch.matmul(anchor_feature, contrast_feature.T), self.temperature
        )

        # for numerical stability, doesn't affect any values ahead
        logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
        logits = anchor_dot_contrast - logits_max.detach()

        # tile mask
        pos_mask = pos_mask.repeat(anchor_count, contrast_count)

        # mask-out self-contrast cases
        logits_mask = torch.scatter(
            torch.ones_like(pos_mask),
            1,
            torch.arange(batch_size * anchor_count).view(-1, 1).to(device),
            0,
        )
        # This is just an inverted identity matrix
        # assert logits_mask.cpu() == (torch.eye(logits_mask.shape[0]) == 0).int()
        pos_mask = pos_mask * logits_mask

        # compute log_prob
        exp_logits = torch.exp(logits) * logits_mask

        if self.formulation == "custom":
            negs_mask = (pos_mask == 0).int() * logits_mask
            negs_sum = (exp_logits * negs_mask).sum(dim=-1, keepdim=True)
            denominator = negs_sum + exp_logits * pos_mask
            log_prob = logits - torch.log(denominator.sum(1, keepdim=True))
        else:
            assert self.formulation == "normal"
            log_prob = logits - torch.log(exp_logits.sum(1, keepdim=True))

        # re-scaling rephrasings
        scl_mask_rescale_factor = registry.scl_mask_rescale_factor
        if scl_mask_rescale_factor > 0:
            secondary_mask = (
                torch.eye(batch_size, device=pos_mask.device)
                .repeat(anchor_count, contrast_count)
                .fill_diagonal_(0)
            )
            secondary_mask = secondary_mask * scl_mask_rescale_factor
            secondary_mask[secondary_mask == 0] = 1
            pos_mask = pos_mask * secondary_mask

        # compute mean of log-likelihood over positive
        mean_log_prob_pos = (pos_mask * log_prob).sum(1) / torch.max(
            pos_mask.sum(1), torch.ones(1).to(pos_mask.device)
        )

        # loss
        loss = -(self.temperature / self.base_temperature) * mean_log_prob_pos
        loss = loss.view(anchor_count, batch_size).mean()

        return loss, -1


LossMap = {
    "BCEWithLogitLoss": torch.nn.BCEWithLogitsLoss(reduction="mean"),
    "SCLLoss": ScaledSupConLoss(
        temperature=registry.get("temperature", 0.5),
        formulation=registry.get("scl_formulation", "normal"),
        base_temperature=registry.get("base_temperature", 0.07),
    ),  # using the default parameter setting
}


def ce_loss(batch_dict, device, val_run=False, revqa_eval=False, split="revqa"):
    if len(batch_dict) == 2 and not val_run:
        # train time
        vil_preds = torch.cat(
            [batch_dict[0]["vil_prediction"], batch_dict[1]["vil_prediction"]], dim=0
        )
        vil_targets = torch.cat(
            [batch_dict[0]["target"], batch_dict[1]["target"]], dim=0
        )
    else:
        if len(batch_dict) == 1:
            batch_dict = batch_dict[0]
        # validation time
        vil_preds = batch_dict["vil_prediction"]
        vil_targets = batch_dict["target"]

    vl_loss = LossMap["BCEWithLogitLoss"](vil_preds, vil_targets)
    vl_loss = vl_loss.mean() * vil_targets.size(1)
    batch_scores = compute_score_with_logits(vil_preds, vil_targets, device)
    batch_score = batch_scores.sum() / len(vil_preds)

    if isinstance(batch_dict, dict):
        # fill the scores for each question into the batch-dict
        batch_dict["vqa_scores"] = batch_scores.sum(dim=-1).tolist()

    # calculate consistency scores during validation
    if revqa_eval:
        # add vqa-scores for each bin
        for idx, qid in enumerate(batch_dict["question_id"].tolist()):
            min_qid = registry[f"question_rephrase_dict_{split}"][qid]
            vqa_score = batch_dict["vqa_scores"][idx]
            bins_key = "revqa_bins" if split in ["revqa", "val"] else "revqa_bt_bins"
            registry[bins_key][min_qid].append((qid, vqa_score))

    return vl_loss, batch_score


def compute_score_with_logits(logits, labels, device):
    logits = torch.max(logits, 1)[1].data  # argmax
    one_hots = torch.zeros(*labels.size())

    if device.type != "cpu":
        one_hots = one_hots.cuda()

    one_hots.scatter_(1, logits.view(-1, 1), 1)
    scores = one_hots * labels
    return scores
