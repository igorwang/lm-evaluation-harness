# -*- coding: utf-8 -*-
"""
-------------------------------------------------
   File Name：     openai_utils
   Description :
   Author :       igorwang
   date：          24/1/2024
-------------------------------------------------
   Change Activity:
                   24/1/2024:
-------------------------------------------------
"""

from openai import OpenAI

client = OpenAI(api_key='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')

response = client.chat.completions.create(
    model="gpt-4-1106-preview",
    messages=[
        {"role": "system", "content": "You are a helpful assistant."},
        {"role": "user",
         "content": "Evaluate the collective function of references cited together in a given text segment.\n Classify their combined role into one of the following categories: 'method', 'background', or 'result'.\n First, list CLUES (i.e., keywords, phrases, contextual information, semantic relations, semantic meaning, tones, references) that support the function determination of input.\n Second, deduce the diagnostic REASONING process from premises (i.e., clues, input) to analyze the function of these references. Determine if they are used to describe the methods employed in the study (procedures, techniques, or approaches), provide background information (theoretical framework, literature review, historical context), or present results (outcomes, findings, data from the study or previous studies).\n Third, based on the identified clues and the diagnostic reasoning process, categorize the collective function of the references into one  categories: 'method', 'background', or 'result'. Conclude the analysis by clearly stating the chosen category in the JSON format: {\"category\": \"label\"}.\n Please ensure that the response MUST includes JSON format conclusion : {\"category\": \"label\"}.\n\nText:Several instruments that more specifically address patient-reported outcomes following gastrectomy are now available, albeit in Japanese [23, 24], and studies using these instruments that have shown some noteworthy results are now starting to be published [25–27].\nCategory:"},
    ]
)


# print(response.json()['choices'][0]['message']['content'])