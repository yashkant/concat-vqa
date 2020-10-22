from itertools import combinations

from tools.registry import registry


def get_consistency_score(results=None, bins_scores=False, bins_key="revqa_bins"):

    # bin the vqa-scores
    revqa_bins_scores = {}

    # vqa-score of all the samples
    total_vqa_scores = []

    for key, value in registry[bins_key].items():

        question_ids, vqa_scores = list(zip(*value))

        if results is not None:
            answers = [results[qid]["answer"] for qid in question_ids]
        else:
            answers = []

        k_values = range(1, 1 + len(value))
        assert len(value) <= 5
        revqa_bins_scores[key] = {
            "vqa_scores": value,
            "question_ids": question_ids,
            "answers": answers,
        }

        total_vqa_scores.extend(value)
        assert sum(vqa_scores) <= len(vqa_scores)

        # for subsets of size = k, check VQA accuracy
        for k_value in k_values:
            value_subsets = list(combinations(vqa_scores, k_value))
            value_subset_scores = []

            # this loop is causing problems!
            for subset in value_subsets:
                if 0.0 not in subset:
                    value_subset_scores.append(1.0)
                else:
                    value_subset_scores.append(0.0)
            revqa_bins_scores[key][k_value] = sum(value_subset_scores) / len(
                value_subsets
            )

    result_dict = {}

    # Consistency Score Calculation
    max_k = 4
    for k_value in range(1, max_k + 1):
        scores = []
        for key, rbs in revqa_bins_scores.items():
            # only consider questions that have all the rephrasings available
            if max_k in rbs:
                scores.append(rbs[k_value])
        add_str = "_bt" if bins_key == "revqa_bt_bins" else ""
        result_dict[str(k_value) + add_str] = sum(scores) / len(scores)
        result_dict[f"len_{k_value}" + add_str] = len(scores)

    if bins_scores:
        return result_dict, revqa_bins_scores

    return result_dict
