import sys

import tqdm
import numpy as np
import torch

from dataloader import FieldBatchDataLoader


METRIC_KEYS = [
    "n_batches", "loss",
    "correct_words", "total_words", "accuracy_words",
    "correct", "total", "accuracy"
]

def initialize_metrics():
    return {key: 0 for key in METRIC_KEYS}

def update_metrics(metrics, batch_output, batch_labels, task=None, with_word_metrics=True):
    # TO_DO: `n_batches` for all tasks
    n_batches = metrics["n_batches"]
    for key, value in batch_output.items():
        if "loss" in key:
            metrics[key] = (metrics.get(key, 0.0) * n_batches + value.item()) / (n_batches + 1)
    # TO_DO: `metrics` for all tasks
    # TO_DO: calculate `correct` and `total` outside for all tasks
    are_equal = (batch_output["labels"] == batch_labels).cpu().numpy()
    non_pad_mask = (batch_labels > 0).cpu().numpy().astype("int")
    metrics["correct"] += (are_equal * non_pad_mask).astype("int").sum()
    metrics["total"] += non_pad_mask.sum()
    if with_word_metrics:
        error_mask = (~are_equal) * non_pad_mask
        metrics["correct_words"] += (are_equal.shape[0] - error_mask.max(axis=-1).sum())
        metrics["total_words"] += are_equal.shape[0]


def do_epoch(model, dataloader, mode="validate", epoch=1, notebook=True,
             task_answer_map=None, with_word_metrics=True):
    task_answer_map = task_answer_map or dict()
    metrics = initialize_metrics()
    func = model.train_on_batch if mode == "train" else model.validate_on_batch
    if notebook:
        progress_bar = tqdm.notebook.tqdm(dataloader, leave=True)
    else:
        progress_bar = tqdm.tqdm(dataloader, leave=True, ncols=200)
    progress_bar.set_description(f"{mode}, epoch={epoch}")
    for batch in progress_bar:
        task = batch.get("task")
        answer_field = task_answer_map.get(task, "y")
        # нужно задать answer_field
        batch_answers = batch.get(answer_field)
        batch_output = func(batch, batch_answers, task=task, mask=batch.get("mask"))
        if "task" not in batch or task == "morpheme":
            update_metrics(metrics, batch_output, batch_answers, with_word_metrics=with_word_metrics)
        # update_metrics(metrics, batch_output, batch_answers, task=task)
        postfix = {key: round(value, 4) for key, value in metrics.items() if "loss" in key}
        for metric, value in metrics.items():
            if "correct" in metric:
                total_value = metrics[metric.replace("correct", "total", 1)]
                acc = value / max(total_value, 1)
                acc_metric = metric.replace("correct", "accuracy", 1)
                metrics[acc_metric] = acc
                postfix[acc_metric] = round(100*acc, 2)
        progress_bar.set_postfix(postfix)
    return metrics


def predict_with_model(model, dataset):
    model.eval()
    dataloader = FieldBatchDataLoader(dataset, device=model.device)
    answer = [None] * len(dataset)
    for batch in dataloader:
        with torch.no_grad():
            batch_answer = model(**batch)
        probs = batch_answer.cpu().numpy()
        labels = probs.argmax(axis=-1)
        for i, label, curr_probs in zip(batch["indexes"], labels, probs):
            length = len(dataset.data[i]["word"])
            curr_labels = "".join(np.take(dataset.labels_, label)[:length])
            answer[i] = {"labels": curr_labels, "probs": curr_probs[:length]}
    return answer


def extract_BIO_boundaries(targets, measure_last=True):
    answer = []
    for i, label in enumerate(targets[1:]):
        if label[0] == "B":
            answer.append(i)
    if measure_last:
        answer.append(len(targets) - 1)
    return answer


def measure_quality_BIO(targets, predicted_targets, english_metrics=False, measure_last=True):
    """
    targets: метки корректных ответов
    predicted_targets: метки предсказанных ответов
    Возвращает словарь со значениями основных метрик
    """
    if "-" not in targets[0]:
        targets = [[x + "-None" for x in elem] for elem in targets]
    if "-" not in predicted_targets[0]:
        predicted_targets = [[x + "-None" for x in elem] for elem in predicted_targets]
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    corr_words = 0
    for corr, pred in zip(targets, predicted_targets):
        boundaries = extract_BIO_boundaries(corr, measure_last=measure_last)
        pred_boundaries = extract_BIO_boundaries(pred, measure_last=measure_last)
        #         print(corr, pred, boundaries, pred_boundaries)
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x == y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
    metrics = ["Точность", "Полнота", "F1-мера", "Корректность", "Точность по словам"]
    if english_metrics:
        metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP + FP), TP / (TP + FN), TP / (TP + 0.5 * (FP + FN)),
               equal / total, corr_words / len(targets)]
    answer = list(zip(metrics, results))
    return answer


def measure_quality(targets, predicted_targets, english_metrics=False, measure_last=True):
    """
    targets: метки корректных ответов
    predicted_targets: метки предсказанных ответов
    Возвращает словарь со значениями основных метрик
    """
    if "-" not in targets[0][0]:
        targets = [[x + "-None" for x in elem] for elem in targets]
    if "-" not in predicted_targets[0]:
        predicted_targets = [[x + "-None" for x in elem] for elem in predicted_targets]
    TP, FP, FN, equal, total = 0, 0, 0, 0, 0
    SE = ['{}-{}'.format(x, y) for x in "SE" for y in ["ROOT", "PREF", "SUFF", "END", "LINK", "None"]]
    # SE = ['S-ROOT', 'S-PREF', 'S-SUFF', 'S-END', 'S-LINK', 'E-ROOT', 'E-PREF', 'E-SUFF', 'E-END']
    corr_words = 0
    for corr, pred in zip(targets, predicted_targets):
        corr_len = len(corr) + int(measure_last) - 1
        pred_len = len(pred) + int(measure_last) - 1
        boundaries = [i for i in range(corr_len) if corr[i] in SE]
        pred_boundaries = [i for i in range(pred_len) if pred[i] in SE]
        common = [x for x in boundaries if x in pred_boundaries]
        TP += len(common)
        FN += len(boundaries) - len(common)
        FP += len(pred_boundaries) - len(common)
        equal += sum(int(x == y) for x, y in zip(corr, pred))
        total += len(corr)
        corr_words += (corr == pred)
    metrics = ["Точность", "Полнота", "F1-мера", "Корректность", "Точность по словам"]
    if english_metrics:
        metrics = ["Precision", "Recall", "F1", "Accuracy", "Word accuracy"]
    results = [TP / (TP + FP), TP / (TP + FN), TP / (TP + 0.5 * (FP + FN)),
               equal / total, corr_words / len(targets)]
    answer = dict(zip(metrics, results))
    return answer


def normalize_target(targets):
    answer = []
    targets = [(x if "-" in x else x + "-None") for x in targets] + ["S-new"]
    state = "E-None"
    for i, label in enumerate(targets[:-1]):
        label, morph = label.split("-")
        next_label, next_morph = targets[i + 1].split("-")
        if state[0] == "E":
            is_last_letter = (morph != next_morph) or next_label in "BS"
            if is_last_letter:
                answer.append(f"S-{morph}")
                state = f"E-{morph}"
            else:
                answer.append(f"B-{morph}")
                state = f"M-{morph}"
        else:
            is_last_letter = (morph != next_morph) or next_label in "BS"
            if is_last_letter:
                answer.append(f"E-{morph}")
                state = f"E-{morph}"
            else:
                answer.append(f"M-{morph}")
                state = f"M-{morph}"
    return answer