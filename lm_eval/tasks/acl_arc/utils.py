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
from lm_eval.api.registry import register_metric, register_aggregation
import numpy as np


def f1(items, average='macro'):
    unzipped_list = list(zip(*items))
    golds = unzipped_list[0]
    preds = unzipped_list[1]
    fscore = sklearn.metrics.f1_score(golds, preds, average='macro')
    return np.max(fscore)


@register_aggregation('f1_macro')
def f1_macro(items):
    return f1(items, average='macro')


@register_aggregation('f1_micro')
def f1_micro(items):
    return f1(items, average='micro')


@register_aggregation('f1_weighted')
def f1_weighted(items):
    return f1(items, average='weighted')


def process_docs(dataset: datasets.Dataset):
    def _process(doc):
        return doc

    return dataset.map(_process)


def process_results(doc, results):
    import re
    intents = ['background', 'uses', 'compares', 'motivation', 'continuation', 'future', 'unknow']
    pattern = '|'.join(intents)
    gold = doc['intent']
    pred = re.findall(pattern, results[0].lower())
    pred = pred[0] if pred else "unknow"
    pred = intents.index(pred)
    return {"f1": (gold, pred), "acc": 1 if gold == pred else 0}
