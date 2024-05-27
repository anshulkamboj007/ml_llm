import pandas as pd
import numpy as np

from sklearn.feature_extraction.text import CountVectorizer
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix,accuracy_score

from collections import Counter

import torch
import torch.nn as nn
from torch.utils.data import Dataset,DataLoader

import seaborn as sns

df=pd.read_csv(r'C:\Users\anshul\Desktop\ML_LLM\ml_llm\nlp\data\Tweets.csv').dropna()

df['class']

print("done")