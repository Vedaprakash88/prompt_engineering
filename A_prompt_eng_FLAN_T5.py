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


################################################################################################################################################

# Summarization Task

################################################################################################################################################

# Define a prompt

prompt = "summarize in 200 characters: These platforms offer varying strengths in data handling, AI capabilities, \
    and scalability. For aerospace companies generating diverse, high-volume data, the ideal platform should excel \
    in comprehensive data integration, robust AI/ML functionalities, and scalability to handle massive datasets. Key \
    considerations include the ability to process complex data types, support for advanced analytics, and seamless \
    integration with industry-specific tools. While each platform has its merits, the most suitable choice depends on \
    specific company needs, existing infrastructure, and long-term data strategy. Factors such as data processing power, \
    AI model management, ecosystem compatibility, and security features play crucial roles in the decision-making process. \
    The optimal platform will provide a versatile, powerful solution capable of supporting various data formats, processing \
    needs, and AI workloads in the technologically advanced and data-intensive aerospace environment."

# Generate tokens (Inputs for the model)

inputs = tokenizer(prompt, return_tensors="tf", max_length=512, truncation=True, padding=True)

# Generate outputs

outputs = model.generate(inputs["input_ids"], max_length = 150,  num_beams=5, early_stopping=True)

# Decode and print

print(tokenizer.decode(outputs[0], skip_special_tokens=True))

################################# End of Summarization ############################################################################################


###################################################################################################################################################

# Translation Task

###################################################################################################################################################

# Define prompt for translation

tranlation_prompt = "translate this from English to German: Germany and India are great countries and have rich history"
find_article = "Find German Article (die, der, or das) for the noun: Kühlschrank"

# Prepare inputs

translation_inputs = tokenizer(find_article, return_tensors="tf", max_length=512, truncation=True, padding=True)

# Generate outputs

tranlation_outputs = model.generate(translation_inputs["input_ids"], max_length=40, num_beams=5, early_stopping=True)

# Decode and display the translation

print(tokenizer.decode(tranlation_outputs[0], skip_special_tokens=True))

################################# End of Translation ##############################################################################################


###################################################################################################################################################

# Question and Answers

###################################################################################################################################################

# context and question
context_question = "The great wall of china is over 13000 miles long. \
                    Question: how long is great wall of china?"

# Generate inputs

question_inputs = tokenizer(context_question, return_tensors="tf", max_length=512, truncation=True, padding=True)

# generate outputs
question_outputs = model.generate(question_inputs["input_ids"], max_length=50, num_beams=5, early_stopping=True)

# decode and print the answer

print(tokenizer.decode(question_outputs[0], skip_special_tokens=True))
