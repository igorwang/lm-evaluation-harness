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


def process_docs(dataset: datasets.Dataset):
    def _process(doc):
        new_doc = {}
        new_doc['sectionName'] = doc['sectionName']
        new_doc['label'] = doc['label']
        new_doc['string'] = doc['string']
        return new_doc

    return dataset.map(_process)


def process_results(doc, results):
    import re
    intents = ['method', 'background', 'result', 'unknown']
    pattern = '|'.join(intents)
    gold = doc['label']
    pred = re.findall(pattern, results[0].lower())
    pred = pred[0] if pred else "unknown"
    pred = intents.index(pred)
    return {"f1": (gold, pred), "acc": 1 if gold == pred else 0}
