# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     utils
   Description :
   Author :       igorwang
   date：          14/12/2023
-------------------------------------------------
   Change Activity:
                   14/12/2023:
-------------------------------------------------
"""
import datasets
import sklearn.metrics
import numpy as np
from lm_eval.api.registry import register_aggregation, register_metric


def f1(items, average='macro'):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds, average='macro')
    return np.max(fscore)


def f1_macro(items):
    return f1(items, average='macro')


def f1_micro(items):
    return f1(items, average='micro')


def f1_weighted(items):
    return f1(items, average='weighted')


def process_docs(dataset: datasets.Dataset):
    def _process(doc):
        new_doc = {}
        new_doc['sectionName'] = doc['sectionName']
        new_doc['label'] = doc['label']
        new_doc['string'] = doc['string']
        # new_doc['fewshots_1'] = doc.get('fewshots_1')
        # new_doc['fewshots_5'] = doc.get('fewshots_5')
        # new_doc['fewshots_10'] = doc.get('fewshots_10')
        return new_doc

    return dataset.map(_process)


def process_results(doc, results):
    import re
    resp = results[0].lower()
    labels = ['method', 'background', 'result']
    pattern = '|'.join(labels)
    # 查找{}
    pred_label = 'unk'
    formatted_resp = re.search('\{.*?\}', resp)
    if formatted_resp:
        pred = re.findall(pattern, formatted_resp.group())
        if pred:
            pred_label = pred[0]
    else:
        pred = re.findall(pattern, resp)
        if pred:
            pred_label = max(set(pred), key=pred.count)
    gold = labels[doc['label']]
    result = {
        "f1_macro": (gold, pred_label),
        "f1_micro": (gold, pred_label),
        "acc": 1 if gold == pred else 0}
    return result
