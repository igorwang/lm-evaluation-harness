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

        return doc

    return dataset.map(_process)
