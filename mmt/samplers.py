import itertools
import json
import logging
import multiprocessing as mp
import os
import pickle as cPickle
import random
import time
from collections import Counter, defaultdict
from copy import deepcopy
from itertools import cycle

import numpy as np
import torch
from easydict import EasyDict as edict
from torch.utils.data.sampler import Sampler
from tqdm import tqdm

from tools.registry import registry

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name


class RandomSampler(Sampler):
    r"""Samples elements randomly. If without replacement, then sample from a shuffled dataset.
    If with replacement, then user can specify ``num_samples`` to draw.

    Arguments:
        data_source (Dataset): dataset to sample from
        num_samples (int): number of samples to draw, default=len(dataset)
        replacement (bool): samples are drawn with replacement if ``True``, default=False

    __iter__() is called after each epoch to get batch indices
    """

    def __init__(self, data_source, replacement=False, num_samples=None):
        self.data_source = data_source
        self.replacement = replacement
        self.num_samples = num_samples

        if self.num_samples is not None and replacement is False:
            raise ValueError(
                "With replacement=False, num_samples should not be specified, "
                "since a random permute will be performed."
            )

        if self.num_samples is None:
            self.num_samples = len(self.data_source)

        if not isinstance(self.num_samples, int) or self.num_samples <= 0:
            raise ValueError(
                "num_samples should be a positive integeral "
                "value, but got num_samples={}".format(self.num_samples)
            )
        if not isinstance(self.replacement, bool):
            raise ValueError(
                "replacement should be a boolean value, but got "
                "replacement={}".format(self.replacement)
            )

    def __iter__(self):
        n = len(self.data_source)
        if self.replacement:
            return iter(
                torch.randint(
                    high=n, size=(self.num_samples,), dtype=torch.int64
                ).tolist()
            )
        return iter(torch.randperm(n).tolist())

    def __len__(self):
        return len(self.data_source)


class ContrastiveSampler(Sampler):
    """
    Sample Contrastive Batches for Scaled Supervised Contrastive Loss.
    """

    def __init__(self, data_source, task_cfg, split="train"):
        self.data_source = data_source
        self.batch_size = task_cfg["batch_size"]

        if "trainval" in split:
            self.split = "trainval"
        elif "train" in split:
            self.split = "train"
        else:
            raise ValueError

        self.arg_split = split
        self.task_cfg = task_cfg
        self.epoch_idx = int(1e10)
        self.epochs = []
        self.num_positives = self.task_cfg.get("num_positives", -1)
        if self.num_positives > 0:
            self.read_annotations()
        self.bin_ans_threshold = self.task_cfg.get("bin_ans_threshold", None)
        self.freq_ans_threshold = self.task_cfg.get("freq_ans_threshold", None)
        self.iter_count = 0
        logger.info(f"Use GT answers is: {self.task_cfg.get('use_gt_answer', False)}")
        self.entries = data_source.entries
        # map from question-id -> entry-idx
        self.question_map = data_source.question_map
        self.load_hard_negatives()

    def load_hard_negatives(self):
        """Load negatives of type `Image` and `Question` """
        negs_path = "data-release/negative-files/fil_{}_question_negs.pkl".format(
            self.split
        )
        logger.info(f"Loading Negatives from: {negs_path}")
        assert os.path.exists(negs_path)
        self.negs_data = cPickle.load(open(negs_path, "rb"))
        self.negs_index_dict = {}
        for idx, qid in enumerate(self.negs_data["qids"]):
            self.negs_index_dict[qid] = idx

    def add_to_answer_map(self, entry_map_key):
        """ Helper function to generate answer -> question-ids  """
        total_answers = len(self.qid_ans_dict[entry_map_key])

        for (ans_label, freq) in self.qid_ans_dict[entry_map_key]:
            if freq >= self.freq_ans_threshold:

                if ans_label in self.answer_map:
                    self.answer_map[ans_label].append((entry_map_key, freq))
                else:
                    self.answer_map[ans_label] = [(entry_map_key, freq)]

    def read_annotations(self):
        """Read VQA Annotations to generate question-id -> answer mapping"""
        ann_path = [
            "data-release/splits/v2_mscoco_train2014_annotations.json",
            "data-release/splits/v2_mscoco_val2014_annotations.json",
        ]
        ann_data = (
            json.load(open(ann_path[0]))["annotations"]
            + json.load(open(ann_path[1]))["annotations"]
        )
        self.qid_ans_dict = {}
        for ann in ann_data:
            ans_counter = Counter()
            for ans in ann["answers"]:
                # only consider answers available in vocabulary
                if ans["answer"] in registry.ans2label:
                    ans_counter[registry.ans2label[ans["answer"]]] += 1

            # build dict with qid -> [(ans_label, freq)...]
            self.qid_ans_dict[ann["question_id"]] = ans_counter.most_common()

    def build_maps(self):
        """
        1. Builds Map from min(repharsed_ids) = info
        2. Builds re_bins = items of above dict sorted by ids
        """

        self.entry_map = {}
        self.answer_map = {}
        for entry in tqdm(
            self.entries,
            desc="Building question and answer maps",
            total=len(self.entries),
        ):
            question_ids = list(
                set(deepcopy(entry["rephrasing_ids"]) + [entry["question_id"]])
            )
            question_ids.sort()
            entry_map_key = min(question_ids)
            # skip creating bins for rephrasings
            if entry_map_key in self.entry_map:
                continue
            self.entry_map[entry_map_key] = {
                "question_ids": question_ids,
                "iter_idx": 0,
                "entry_inds": [self.question_map[x] for x in question_ids],
            }
            self.add_to_answer_map(entry_map_key)

        # post-process: remove duplicates, build sampling weights
        for key in tqdm(
            self.answer_map.keys(), desc="Post-processing", total=len(self.answer_map)
        ):
            self.answer_map[key] = list(set(self.answer_map[key]))
            ans_labels, freqs = list(zip(*self.answer_map[key]))
            self.answer_map[key] = (cycle(ans_labels), len(ans_labels))

        self.re_bins = sorted(self.entry_map.items(), key=lambda x: x[0])
        for idx, bin in enumerate(self.re_bins):
            bin[1]["bin_idx"] = idx
            bin[1]["iter_idx"] = cycle(list(range(len(bin[1]["question_ids"]))))

    @staticmethod
    def get_hard_negative(negative_list, batch_bins, entry_map, question_rephrase_dict):
        """ Given a list of negatives return a valid one. """
        if len(negative_list) == 0:
            return -1, True, -1

        for qid in negative_list:
            if qid < 0:
                assert qid == -1
                break

            # handle case when we don't use all the rephrasings
            if qid not in question_rephrase_dict:
                continue

            source_id = question_rephrase_dict[qid]
            item = entry_map[source_id]
            bin_idx = item["bin_idx"]
            if bin_idx not in batch_bins:
                iter_idx = next(item["iter_idx"])
                entry_idx = item["entry_inds"][iter_idx]
                return entry_idx, False, bin_idx

        return -1, True, -1

    def build_hard_batches(self):
        """ Build batches w/ `Random`, `Image` and `Question` negatives. """
        self.build_maps()
        self.re_bins = ContrastiveSampler.shuffle(self.re_bins, 0, len(self.re_bins))
        init_batch_size = self.task_cfg["init_batch_size"]
        neg_type_weights = self.task_cfg["neg_type_weights"]
        assert np.sum(neg_type_weights) == 1.0
        init_pass_bs = init_batch_size + self.num_positives * init_batch_size
        num_batches = int(len(self.entries) / init_batch_size)
        question_rephrase_dict = getattr(
            registry, f"question_rephrase_dict_{self.split}"
        )

        # actual no. of batches to return (for one epoch)
        self.num_batches = int(len(self.entries) / self.batch_size)

        extra_args = edict()
        extra_args.update(
            {
                "num_positives": self.num_positives,
                "bin_ans_threshold": self.bin_ans_threshold,
            }
        )

        _args = [
            self.entry_map,
            self.re_bins,
            self.answer_map,
            self.qid_ans_dict,
            extra_args,
            num_batches if num_batches < 20000 else 20000,
            init_pass_bs,
        ]

        # shuffle bins
        self.re_bins = ContrastiveSampler.shuffle(self.re_bins, 0, len(self.re_bins))

        # add references and positives
        batches, batches_bins = ContrastiveSampler.get_batches(_args)

        # replace w/ original batch-size
        _args[-1] = self.batch_size
        _args += list(
            [neg_type_weights, self.entries, self.negs_data, self.negs_index_dict]
        )

        # shuffle bins
        self.re_bins = ContrastiveSampler.shuffle(self.re_bins, 0, len(self.re_bins))

        # add hard-negatives
        batches, batches_bins = ContrastiveSampler.add_hard_negatives(
            batches, batches_bins, _args, question_rephrase_dict
        )

        num_epochs = int(len(batches) / self.num_batches)
        epochs = []

        # build epochs
        for epoch_idx in range(num_epochs):
            batch_start_idx = epoch_idx * self.num_batches
            batch_end_idx = (epoch_idx + 1) * self.num_batches
            assert batch_end_idx <= len(batches)
            epoch = []
            for batch_idx in range(batch_start_idx, batch_end_idx):
                assert len(batches[batch_idx]) == len(set(batches[batch_idx]))
                epoch.extend(batches[batch_idx])
            epochs.append(epoch)

        self.epoch_idx = 0
        self.epochs = epochs

    @staticmethod
    def shuffle(array, start_idx, end_idx):
        """ Shuffle elements in a given subset of array. """
        np.random.shuffle(array[start_idx:end_idx])
        for i, item in enumerate(array[start_idx:end_idx]):
            item[1]["bin_idx"] = i + start_idx
        ContrastiveSampler.assert_bins(array)
        return array

    @staticmethod
    def get_batches(args):
        (
            entry_map,
            re_bins,
            answer_map,
            qid_ans_dict,
            extra_args,
            num_batches,
            batch_size,
        ) = args
        batches = []
        batches_bins = []
        num_positives = extra_args.num_positives
        add_positives = num_positives > 0
        bins_iterator = cycle(range(len(re_bins)))

        for _ in tqdm(
            zip(range(num_batches)), total=num_batches, desc="Build Batches [References and Intra-class Positives]"
        ):

            # start building a batch
            batch_inds = []
            batch_bins = []
            while True:
                bin_idx = next(bins_iterator)

                # to account for bins-used by positive sampler
                if bin_idx in batch_bins:
                    continue

                # pick the value from (key,value) tuple
                item = re_bins[bin_idx][1]

                # randomly pick one entry from the bin
                iter_idx = next(item["iter_idx"])
                entry_idx = item["entry_inds"][iter_idx]
                batch_inds.append(entry_idx)
                batch_bins.append(bin_idx)

                if add_positives:
                    # only add the needed amount
                    num_pos = min(num_positives, batch_size - len(batch_inds))
                    ContrastiveSampler.add_positives(
                        re_bins,
                        entry_map,
                        qid_ans_dict,
                        answer_map,
                        bin_idx,
                        num_pos,
                        batch_inds,
                        batch_bins,
                        extra_args,
                    )
                if len(batch_inds) == batch_size:
                    break

            assert len(batch_bins) == len(set(batch_bins)) == batch_size
            batches.append(batch_inds)
            batches_bins.append(batch_bins)

        return batches, batches_bins

    @staticmethod
    def add_hard_negatives(batches, batches_bins, args, question_rephrase_dict):
        """ Given batch of only reference samples and positives, add negatives.  """
        (
            entry_map,
            re_bins,
            answer_map,
            qid_ans_dict,
            extra_args,
            num_batches,
            batch_size,
            neg_type_weights,
            entries,
            negs_data,
            negs_index_dict,
        ) = args

        bins_iterator = cycle(range(len(re_bins)))

        for batch_inds, batch_bins in tqdm(
            zip(batches, batches_bins), total=len(batches), desc="Build Batches [Negatives]"
        ):
            batch_inds_iter = cycle(batch_inds)

            while True:
                neg_choice = np.random.choice(
                    ["image_neg", "question_neg", "random"], p=neg_type_weights
                )
                passed = False

                if neg_choice in ["image_neg"]:
                    entry_idx = next(batch_inds_iter)
                    question_id = entries[entry_idx]["question_id"]
                    negs_idx = negs_index_dict[question_id]
                    negatives_list = negs_data["same_image_questions_neg"][negs_idx]
                    # add better negatives
                    (
                        neg_entry_idx,
                        passed,
                        bin_idx,
                    ) = ContrastiveSampler.get_hard_negative(
                        negatives_list, batch_bins, entry_map, question_rephrase_dict
                    )
                    if not passed:
                        batch_inds.append(neg_entry_idx)
                        batch_bins.append(bin_idx)

                if neg_choice in ["question_neg"] or passed:
                    entry_idx = next(batch_inds_iter)
                    entry = entries[entry_idx]
                    question_id = entry["question_id"]

                    if "top_k_questions_neg" in entry:
                        negatives_list = entry["top_k_questions_neg"]
                    else:
                        negs_idx = negs_index_dict[question_id]
                        negatives_list = negs_data["question_negs"][negs_idx]

                    # add better negatives
                    (
                        neg_entry_idx,
                        passed,
                        bin_idx,
                    ) = ContrastiveSampler.get_hard_negative(
                        negatives_list, batch_bins, entry_map, question_rephrase_dict
                    )
                    if not passed:
                        batch_inds.append(neg_entry_idx)
                        batch_bins.append(bin_idx)

                if neg_choice == "random" or passed:
                    while True:
                        bin_idx = next(bins_iterator)
                        # to account for bins-used by positive sampler
                        if bin_idx in batch_bins:
                            continue
                        # pick the value from (key,value) tuple
                        item = re_bins[bin_idx][1]
                        # randomly pick one entry from the bin
                        iter_idx = next(item["iter_idx"])
                        entry_idx = item["entry_inds"][iter_idx]
                        batch_inds.append(entry_idx)
                        batch_bins.append(bin_idx)
                        break

                if len(batch_inds) == batch_size:
                    assert len(batch_bins) == len(set(batch_bins)) == batch_size
                    break

        return batches, batches_bins

    @staticmethod
    def assert_bins(array):
        for i, item in enumerate(array):
            assert item[1]["bin_idx"] == i

    @staticmethod
    def add_positives(
        re_bins,
        entry_map,
        qid_ans_dict,
        answer_map,
        bin_idx,
        num_positives,
        batch_inds,
        batch_bins,
        extra_args,
    ):
        """Add intra-class positives given references"""

        if num_positives <= 0:
            return

        # sample bin-answer to select positive from
        bin_min_qid = min(re_bins[bin_idx][1]["question_ids"])
        bin_answers = qid_ans_dict[bin_min_qid]

        filtered_bin_answers = []
        filtered_bin_answers_weights = []

        for ans, freq in bin_answers:
            if freq >= extra_args.bin_ans_threshold and int(ans) in answer_map:
                filtered_bin_answers.append(ans)
                filtered_bin_answers_weights.append(freq)

        if len(filtered_bin_answers) <= 0:
            return

        filtered_bin_answers_weights = np.array(filtered_bin_answers_weights) / sum(
            filtered_bin_answers_weights
        )
        answer = int(
            np.random.choice(filtered_bin_answers, 1, p=filtered_bin_answers_weights)
        )

        count_pos = 0
        qids_iter, qids_len = answer_map[int(answer)]
        start_qid = next(qids_iter)

        # get corresponding bins and update batch
        qid = start_qid
        while True:
            item = entry_map[qid]

            # skip if already present in batch
            if item["bin_idx"] in batch_bins:
                qid = next(qids_iter)
                # we have exhausted all positives
                if qid == start_qid:
                    break

                continue
            # this condition breaks the loop after needed positives in such case
            if count_pos == num_positives:
                break
            batch_bins.append(item["bin_idx"])
            iter_idx = next(item["iter_idx"])
            entry_idx = item["entry_inds"][iter_idx]
            batch_inds.append(entry_idx)
            count_pos += 1
            qid = next(qids_iter)
            # we have exhausted all positives
            if qid == start_qid:
                break

    def __iter__(self):
        # if epochs are exhausted, replenish
        if self.epoch_idx >= len(self.epochs):
            self.build_hard_batches()

        epoch_indices = self.epochs[self.epoch_idx]
        self.epoch_idx += 1
        logger.info(
            f"No. of Unique Samples: {len(set(epoch_indices))} / {len(epoch_indices)}"
        )
        self.iter_count += 1
        return iter(epoch_indices)

    def __len__(self):
        return len(self.data_source)
