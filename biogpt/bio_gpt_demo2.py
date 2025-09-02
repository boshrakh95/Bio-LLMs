# Extended use of BioGPT 
# https://huggingface.co/docs/transformers/v4.56.0/en/model_doc/biogpt 
# Boshra
# 02 September 2025

# NOTE 1: 
# transformers.BioGptModel: The bare Biogpt Model outputting raw hidden-states without any specific head on top.
# transformers.BioGptForCausalLM: Biogpt Model with a language modeling head on top (linear layer with weights tied to the input embeddings).
# transformers.BioGptForTokenClassification: Biogpt Model with a token classification head on top (linear layer on top of the hidden states) e.g. for Named-Entity-Recognition (NER) tasks.
# transformers.BioGptForSequenceClassification: Biogpt Model with a sequence classification head on top e.g. for text classification tasks. It uses the last token in order to do the classification, as other causal models (e.g. GPT-2) do.
# NOTE 2:
# BioGptForTokenClassification, BioGptForSequenceClassification need fine-tuning on a labeled dataset to be effective.

from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig
from transformers import BioGptModel, BioGptConfig, BioGptForTokenClassification, BioGptForSequenceClassification
from transformers import set_seed, pipeline
import torch
import torch.nn.functional as F

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

# -------------------------------
# Github Examples (modified):
# -------------------------------

tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = AutoModelForCausalLM.from_pretrained(
    "microsoft/biogpt",
    dtype=torch.float16,
    device_map="auto",
    attn_implementation="sdpa"
)

# 1- Generate text using AutoModel
input_text = "Ibuprofen is best used for"
inputs = tokenizer(input_text, return_tensors="pt").to(model.device)

with torch.no_grad():
    generated_ids = model.generate(**inputs, max_length=50)
    
output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(output)

# 2- Generate text using pipeline
generator = pipeline(
    task="text-generation",
    model="microsoft/biogpt",
    dtype=torch.float16,
    device=0,
)
result = generator("Ibuprofen is best used for", truncation=True, max_length=50, do_sample=True)[0]["generated_text"]
print(result)

# 3- Quantization (Not possible on Apple Silicon)
# Quantization reduces the memory burden of large models by representing the weights in a lower precision.
# Use bitsandbytes to only quantize the weights to 4-bit precision.
# bnb_config = BitsAndBytesConfig(
#     load_in_4bit=True,
#     bnb_4bit_quant_type="nf4",
#     bnb_4bit_compute_dtype=torch.bfloat16,
#     bnb_4bit_use_double_quant=True
# )

# tokenizer = AutoTokenizer.from_pretrained("microsoft/BioGPT-Large")
# model = AutoModelForCausalLM.from_pretrained(
#     "microsoft/BioGPT-Large", 
#     quantization_config=bnb_config,
#     dtype=torch.bfloat16,
#     device_map="auto"
# )

# input_text = "Ibuprofen is best used for"
# inputs = tokenizer(input_text, return_tensors="pt").to(model.device)
# with torch.no_grad():
#     generated_ids = model.generate(**inputs, max_length=50)    
# output = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
# print(output)

# 4- Accessing model configuration
# Initializing a BioGPT microsoft/biogpt style configuration
configuration = BioGptConfig()
# Initializing a model from the microsoft/biogpt style configuration
model = BioGptModel(configuration)
# Accessing the model configuration
configuration = model.config
# Possible labels if using token classification
configuration.id2label
#{0: 'LABEL_0', 1: 'LABEL_1'}

# 5- Token Classification
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForTokenClassification.from_pretrained("microsoft/biogpt")

inputs = tokenizer(
    "HuggingFace is a company based in Paris and New York", add_special_tokens=False, return_tensors="pt"
)

with torch.no_grad():
    logits = model(**inputs).logits

predicted_token_class_ids = logits.argmax(-1)

# Note that tokens are classified rather then input words which means that
# there might be more predicted token classes than words.
# Multiple token classes might account for the same word
predicted_tokens_classes = [model.config.id2label[t.item()] for t in predicted_token_class_ids[0]]
predicted_tokens_classes

# NOTE: The BioGPT token classification model hasn’t been fine-tuned for a specific token classification task yet. 
# That’s why its labels are just generic placeholders: 0: 'LABEL_0' 1: 'LABEL_1'
# This means the model is initialized for token classification, but it doesn’t know any real named entity types until it’s fine-tuned on a dataset.
# So: LABEL_0 and LABEL_1 are just default labels.

# Optional loss calculation (for training)
labels = predicted_token_class_ids
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)

# 6- Sequence Classification (single-label classification)
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_id = logits.argmax().item()
model.config.id2label[predicted_class_id]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", num_labels=num_labels)

labels = torch.tensor([1])
loss = model(**inputs, labels=labels).loss
round(loss.item(), 2)

# NOTE: The model needs fine-tuning for sequence classification tasks.

# 7- Sequence Classification (multi-label classification)
tokenizer = AutoTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForSequenceClassification.from_pretrained("microsoft/biogpt", problem_type="multi_label_classification")

inputs = tokenizer("Hello, my dog is cute", return_tensors="pt")

with torch.no_grad():
    logits = model(**inputs).logits

predicted_class_ids = torch.arange(0, logits.shape[-1])[torch.sigmoid(logits).squeeze(dim=0) > 0.5]

# To train a model on `num_labels` classes, you can pass `num_labels=num_labels` to `.from_pretrained(...)`
num_labels = len(model.config.id2label)
model = BioGptForSequenceClassification.from_pretrained(
    "microsoft/biogpt", num_labels=num_labels, problem_type="multi_label_classification"
)

labels = torch.sum(
    torch.nn.functional.one_hot(predicted_class_ids[None, :].clone(), num_classes=num_labels), dim=1
).to(torch.float)
loss = model(**inputs, labels=labels).loss

# NOTE: The model needs fine-tuning for sequence classification tasks.



