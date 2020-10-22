import json
import logging
import os
from copy import deepcopy

import _pickle as cPickle
import numpy as np
import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from tools.registry import registry

logger = logging.getLogger(__name__)  # pylint: disable=invalid-name
os.environ["HDF5_USE_FILE_LOCKING"] = "FALSE"


def assert_eq(real, expected):
    assert real == expected, "%s (true) vs %s (expected)" % (real, expected)


def create_entry(question, answer):
    answer.pop("image_id")
    answer.pop("question_id")
    entry = {
        "question_id": question["question_id"],
        "image_id": question["image_id"],
        "question": question["question"],
        "answer": answer,
    }
    retain_keys = [
        "rephrasing_ids",
        "top_k_questions",
        "top_k_questions_neg",
        "same_image_questions",
        "same_image_questions_neg",
    ]
    for key in retain_keys:
        if key in question:
            entry[key] = question[key]

    return entry


def filter_aug(questions_list, answers_list):
    questions, answers = [], []
    max_samples = registry.aug_filter["num_rephrasings"]
    sim_threshold = registry.aug_filter["sim_threshold"]
    assert len(questions_list) == len(answers_list)

    for idx, (que_list, ans_list) in tqdm(
        enumerate(zip(questions_list, answers_list)),
        total=len(questions_list),
        desc="Filtering Data",
    ):

        assert len(que_list) == len(ans_list)
        # filter for sim-threshold
        if sim_threshold > 0:
            que_list, ans_list = zip(
                *[
                    (q, a)
                    for q, a in zip(que_list, ans_list)
                    if q["sim_score"] > sim_threshold
                ]
            )

        # filter for max-samples
        if max_samples > 0:
            que_list, ans_list = que_list[:max_samples], ans_list[:max_samples]

        filtered_rephrasing_ids = [que["question_id"] for que in que_list]
        min_qid = min(filtered_rephrasing_ids)
        for que in que_list:
            que["rephrasing_ids"] = sorted(
                [x for x in filtered_rephrasing_ids if x != que["question_id"]]
            )
            if "rephrasing_of" not in que:
                que["rephrasing_of"] = min_qid
            else:
                assert min_qid == que["rephrasing_of"]

        # add them to main list
        questions.extend(que_list)
        answers.extend(ans_list)

    return questions, answers


def rephrasings_dict(split, questions):
    question_rephrase_dict = {}
    for question in questions:
        if "rephrasing_of" in question:
            question_rephrase_dict[question["question_id"]] = question["rephrasing_of"]
        elif "rephrasing_ids" in question:
            min_qid = min(question["rephrasing_ids"] + [question["question_id"]])
            question_rephrase_dict[question["question_id"]] = min_qid
        else:
            question_rephrase_dict[question["question_id"]] = question["question_id"]

    # used in evaluation, hack to set attribute
    from easydict import EasyDict

    super(EasyDict, registry).__setattr__(
        f"question_rephrase_dict_{split}", question_rephrase_dict
    )
    super(EasyDict, registry).__setitem__(
        f"question_rephrase_dict_{split}", question_rephrase_dict
    )
    print(f"Built dictionary: question_rephrase_dict_{split}")


def load_qa(name, sort=True, use_filter=False, set_dict=False):
    split_path_dict = {
        "train_aug": [
            "data-release/splits/questions_train_aug.pkl",
            "data-release/splits/ans_train_aug.pkl",
            "train",
        ],
        "train": [
            "data-release/splits/v2_OpenEnded_mscoco_train2014_questions.json",
            "data-release/splits/train_target.pkl",
            "train",
        ],
        "val": [
            "data-release/splits/v2_OpenEnded_mscoco_val2014_questions.json",
            "data-release/splits/val_target.pkl",
            "val",
        ],
        "val_aug": [
            "data-release/splits/questions_val_aug.pkl",
            "data-release/splits/ans_val_aug.pkl",
            "val",
        ],
        "test": [
            "data-release/splits/v2_OpenEnded_mscoco_test2015_questions.json",
            "",
            "test",
        ],
        "trainval_aug": [
            "data-release/splits/questions_trainval_aug.pkl",
            "data-release/splits/ans_trainval_aug.pkl",
            "trainval",
        ],
        "revqa": [
            "data-release/splits/revqa_total_proc.pkl",
            "data-release/splits/revqa_total_proc_target.pkl",
            "revqa",
        ],
    }
    questions_path, answers_path, split = split_path_dict[name]
    questions = (
        json.load(open(questions_path))
        if questions_path.endswith(".json")
        else cPickle.load(open(questions_path, "rb"))
    )

    if isinstance(questions, dict):
        questions = questions["questions"]

    if name == "test":
        return questions

    answers = cPickle.load(open(answers_path, "rb"))

    if sort:
        questions = sorted(questions, key=lambda x: x["question_id"])
        answers = sorted(answers, key=lambda x: x["question_id"])

    if use_filter:
        questions, answers = filter_aug(questions, answers)

    if set_dict:
        rephrasings_dict(split, questions)

    assert len(questions) == len(answers)
    logger.info(f"Total Samples: {len(questions)} with filtering: {use_filter}")
    return questions, answers


def load_entries(name):
    """Load questions and answers."""

    logger.info(f"Loading Split: {name}")
    if name == "train" or name == "val":
        questions, answers = load_qa(name)
        # if registry.debug:
        #     questions, answers = questions[:40000], answers[:40000]

    elif name in ["train_aug", "val_aug", "trainval_aug"]:
        questions, answers = load_qa(name, sort=False, use_filter=True, set_dict=True)

    elif name == "revqa":
        questions, answers = load_qa(name, sort=False, use_filter=False, set_dict=True)

    elif name == "revqa_bt":
        val_questions, val_answers = load_qa("val_aug", sort=False, use_filter=True)
        dump_path = "data-release/splits/non_overlap_ids.npy"
        non_overlap_ids = np.load(dump_path, allow_pickle=True)
        questions, answers = [], []
        for q, a in zip(val_questions, val_answers):
            if q["rephrasing_of"] in non_overlap_ids:
                questions.append(q)
                answers.append(a)
        rephrasings_dict(name, questions)

    elif name == "trainval":
        questions_train, answers_train = load_qa("train", sort=True)
        questions_val, answers_val = load_qa("val", sort=True)
        questions = questions_train + questions_val[:-3000]
        answers = answers_train + answers_val[:-3000]

    elif name == "minval":
        questions_val, answers_val = load_qa("val", sort=True)
        questions, answers = questions_val[-3000:], answers_val[-3000:]

    elif name == "test":
        questions = load_qa(name)

    else:
        assert False, f"data split {name} is not recognized."

    if "test" in name:
        entries = []
        for question in questions:
            entries.append(question)
    else:
        assert_eq(len(questions), len(answers))
        entries = []

        for question, answer in tqdm(
            zip(questions, answers), total=len(questions), desc="Building Entries"
        ):
            assert_eq(question["question_id"], answer["question_id"])
            assert_eq(question["image_id"], answer["image_id"])
            entries.append(create_entry(question, answer))

    return entries


class VQAClassificationDataset(Dataset):
    def __init__(
        self,
        split,
        image_features_reader,
        tokenizer,
        padding_index=0,
        extra_args=None,
    ):
        """
        Builds self.entries by reading questions and answers and caches them.
        """
        super().__init__()
        self.split = split
        self.ans2label = cPickle.load(
            open("data-release/splits/trainval_ans2label.pkl", "rb")
        )
        self.label2ans = cPickle.load(
            open("data-release/splits/trainval_label2ans.pkl", "rb")
        )
        # attach to registry
        registry.ans2label = self.ans2label
        registry.label2ans = self.label2ans
        self.num_labels = len(self.ans2label)
        self._max_region_num = extra_args["max_region_num"]
        self._max_seq_length = extra_args["max_seq_length"]
        self._image_features_reader = image_features_reader
        self._tokenizer = tokenizer
        self._padding_index = padding_index
        self.extra_args = extra_args

        self.entries = load_entries(split)
        # convert questions to tokens, create masks, segment_ids
        self.tokenize(self._max_seq_length)
        self.tensorize()
        self.mean_read_time = 0.0
        self.num_samples = 0

    def tokenize(self, max_length=16):
        """Tokenizes the questions."""
        self.question_map = {}
        for _idx, entry in enumerate(tqdm(self.entries, desc="Tokenizing...")):
            self.question_map[entry["question_id"]] = _idx
            tokens = self._tokenizer.encode(entry["question"])
            tokens = tokens[: max_length - 2]
            tokens = self._tokenizer.add_special_tokens_single_sentence(tokens)
            segment_ids = [0] * len(tokens)
            input_mask = [1] * len(tokens)

            if len(tokens) < max_length:
                # Note here we pad in front of the sentence
                padding = [self._padding_index] * (max_length - len(tokens))
                tokens = tokens + padding
                input_mask += padding
                segment_ids += padding

            assert_eq(len(tokens), max_length)
            entry["q_token"] = tokens
            entry["q_input_mask"] = input_mask
            entry["q_segment_ids"] = segment_ids

    def tensorize(self):
        for entry in tqdm(self.entries, desc="Tensorizing..."):
            question = torch.from_numpy(np.array(entry["q_token"]))
            entry["q_token"] = question

            q_input_mask = torch.from_numpy(np.array(entry["q_input_mask"]))
            entry["q_input_mask"] = q_input_mask

            q_segment_ids = torch.from_numpy(np.array(entry["q_segment_ids"]))
            entry["q_segment_ids"] = q_segment_ids

            if "test" not in self.split:
                answer = entry["answer"]
                labels = np.array(answer["labels"])
                scores = np.array(answer["scores"], dtype=np.float32)
                if len(labels):
                    labels = torch.from_numpy(labels)
                    scores = torch.from_numpy(scores)
                    entry["answer"]["labels"] = labels
                    entry["answer"]["scores"] = scores
                else:
                    entry["answer"]["labels"] = None
                    entry["answer"]["scores"] = None

    def __getitem__(self, index):
        """
        1. Get image-features/bboxes and image mask (as nump-arrays), tensorize them.
        2. Get question, input_mask, segment_ids and coattention mask
        3. Build target (vocab-dim) with VQA scores scattered at label-indices
        4. Return
        """
        item_dict = {}
        entry = self.entries[index]
        image_id = entry["image_id"]
        question_id = entry["question_id"]
        features, num_boxes, boxes, _ = self._image_features_reader[image_id]
        mix_num_boxes = min(int(num_boxes), self._max_region_num)
        mix_boxes_pad = np.zeros((self._max_region_num, 5))
        mix_features_pad = np.zeros((self._max_region_num, 2048))
        image_mask = [1] * (int(mix_num_boxes))
        while len(image_mask) < self._max_region_num:
            image_mask.append(0)
        mix_boxes_pad[:mix_num_boxes] = boxes[:mix_num_boxes]
        mix_features_pad[:mix_num_boxes] = features[:mix_num_boxes]
        features = torch.tensor(mix_features_pad).float()
        image_mask = torch.tensor(image_mask).long()
        spatials = torch.tensor(mix_boxes_pad).float()
        question = entry["q_token"]
        input_mask = entry["q_input_mask"]
        target = torch.zeros(self.num_labels)

        if "test" not in self.split:
            answer = entry["answer"]
            labels = answer["labels"]
            scores = answer["scores"]
            if labels is not None:
                target.scatter_(0, labels, scores)

        item_dict.update(
            {
                "input_imgs": features,
                "image_mask": image_mask,
                "image_loc": spatials,
                "question_indices": question,
                "question_mask": input_mask,
                "image_id": image_id,
                "question_id": question_id,
                "target": target,
                "mask": torch.tensor(1),
            }
        )

        return_list = [item_dict]

        # if registry.debug:
        #     return return_list

        # don't use while evaluation loop
        if self.extra_args.get("contrastive", None) in [
            "better"
        ] and self.split not in ["minval", "revqa", "test", "val"]:
            num_rep = 2  # number of rephrasing batches
            item_pos_dicts = [deepcopy(item_dict) for _ in range(num_rep - 1)]
            # when there's no rephrasing available send the original
            if len(entry["rephrasing_ids"]) == 0:
                item_dict["mask"] = item_dict["mask"] * 0
                for id in item_pos_dicts:
                    id["mask"] = id["mask"] * 0
                return_list.extend(item_pos_dicts)
                return return_list

            que_ids = np.random.choice(entry["rephrasing_ids"], num_rep - 1)
            pos_entries = [self.entries[self.question_map[qid]] for qid in que_ids]

            for id, pe in zip(item_pos_dicts, pos_entries):
                id.update(
                    {
                        "question_indices": pe["q_token"],
                        "question_mask": pe["q_input_mask"],
                        "question_id": pe["question_id"],
                    }
                )
            return_list.extend(item_pos_dicts)
            return return_list

        return return_list

    def __len__(self):
        return len(self.entries)
