import tensorflow as tf

# For this task, we are using Seq2Seq model, which converts one sequence of text to other.
# Suitable for translation and summarization

# The model is 3.13GB large.

from transformers import TFAutoModelForSeq2SeqLM, AutoTokenizer

# loading the tokenizer and model

tokenizer = AutoTokenizer.from_pretrained("google/flan-t5-large")
model = TFAutoModelForSeq2SeqLM.from_pretrained("google/flan-t5-large")

'''

There are alaways four steps in an LLM project.

- Define a prompt
- Generate tokens (Inputs for the model)
    - pass inputs to model
- Generate Outputs (from the model)
- Decode output and print it


'''


###############################################################################################################################################

# Summarization with Few-Shot Learning

###############################################################################################################################################

# Define an example prompt

few_shot_examples = [
    "summarize : 'Thequick brown fox jumps over the lazy dog. The dog was not amused be the fox's antics.' \
        Summary: 'The fox jumped over the dog, who was not happy.'",
    "Summarize : 'The rain in spain stays mainly in the plan. It was a wet and rainy season in the Spanish plains.' \
        Summary: 'It was a rainy season in the Spanish plains.'"
]

# define the actual prompt
new_prompt = "Summarize : 'Studies show that eating carrots helps improve vision. Carrots contain beta-carotene, \
    a substance that the body converts into vitamin A, crucial for maintaining healthy eyesight"

# combine examples and the prompt

combined_prompt = "\n\n".join(few_shot_examples + [new_prompt])

# Generate tokens (Inputs for the model)

inputs = tokenizer(combined_prompt, return_tensors="tf", max_length=512, truncation=True, padding=True)

# Generate outputs

outputs = model.generate(inputs["input_ids"], max_length = 150,  num_beams=5, early_stopping=True)

# Decode and print

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

################################# End of Summarization #########################################################################################




################################################################################################################################################

# Translation with Few-Shot Learning

################################################################################################################################################

# Define an example prompt

trans_few_shot_examples = [
    "translate English to Spanish: 'The cat sits on the mat.' Translation: 'El gato se sienta en la alfombra.'",
    "translate English to Spanish: 'The sun is shining brightly.' Translaton: 'El sol brilla intensamente'"
]

# define the actual prompt
translation_prompt = "translate English to Spanish: 'Cheese is delicious.'"

# combine examples and the prompt
combined_translation_prompt = "\n\n".join(trans_few_shot_examples + [translation_prompt])

# Generate tokens (Inputs for the model)
translation_inputs = tokenizer(combined_translation_prompt, return_tensors="tf", max_length=512, truncation=True, padding=True)

# Generate outputs
translation_outputs = model.generate(translation_inputs["input_ids"], max_length = 150,  num_beams=5, early_stopping=True)

# Decode and print
print(tokenizer.decode(translation_outputs[0], skip_special_tokens=True))

################################# End of Summarization ############################################################################################



###################################################################################################################################################

# Q&A with In-Context Learninig

###################################################################################################################################################

# Define an example prompt

qna_few_shot_examples = [
    "The great wall of china is over 13000 miles long. Question: 'how long is great wall of china?', answer: 'over 13000 miles long'",
    "The capital of France is Paris. Question: 'What is the capital of France?' answer: 'the capital of france is paris'"
]

# define the actual prompt
qna_prompt = "Mount everest is the highest mountain in the world. question: 'What is the highest mountain in  the world'"

# combine examples and the prompt
combined_qa_prompt = "\n\n".join(qna_few_shot_examples + [qna_prompt])

# Generate tokens (Inputs for the model)
qna_inputs = tokenizer(combined_qa_prompt, return_tensors="tf", max_length=512, truncation=True, padding=True)

# Generate outputs
qna_outputs = model.generate(qna_inputs["input_ids"], max_length = 150,  num_beams=5, early_stopping=True)

# Decode and print
print(tokenizer.decode(qna_outputs[0], skip_special_tokens=True))

################################# End of Summarization ##########################################################################################