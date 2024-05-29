# -*- coding: utf-8 -*-
import sys
sys.path.append('.') # win vscode debug 
sys.path.append('..')
sys.path.append('../..')
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '.'))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))
sys.path.append(os.path.join(os.path.dirname(__file__), '../..'))
import pprint
import paddle
import paddlenlp
from paddlenlp.transformers import ErnieGramModel, ErnieGramTokenizer
from paddlenlp.transformers import ErnieGramForSequenceClassification
from paddlenlp.datasets import MapDataset
from paddlenlp.data import Stack, Pad, Tuple


dialogues = ["你好", "你好，有什么可以帮忙的吗？", "我想查询天气", "请问你想查询哪个城市的天气？"]

tokenizer = ErnieGramTokenizer.from_pretrained('ernie-gram-zh')  
model = ErnieGramModel.from_pretrained('ernie-gram-zh')
# model = ErnieGramForSequenceClassification.from_pretrained('ernie-gram-zh')





