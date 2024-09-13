# For this task, we are using Seq2Seq model, which converts one sequence of text to other.
# Suitable for translation and summarization

# The model is 3.13GB large.

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer
from datasets import load_dataset

# loading the Dailymail dataset

dataset = load_dataset('abisee/cnn_dailymail', '3.0.0')
print(dataset['train'][0])
# loading the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")