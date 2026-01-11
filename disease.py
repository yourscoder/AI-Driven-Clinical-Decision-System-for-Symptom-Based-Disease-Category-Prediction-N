import pandas as pd
import numpy as np
import torch
import spacy
import re

from google.colab import files
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

from transformers import (
    DistilBertTokenizerFast,
    DistilBertForSequenceClassification,
    Trainer,
    TrainingArguments
)

"""# dataset insertion"""

uploaded = files.upload()
df = pd.read_csv(list(uploaded.keys())[0])
