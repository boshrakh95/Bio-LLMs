# Limited but immediate use of BioGPT by Hugging Face Transformers wrapper (load using BioGptForCausalLM and not AutoModelForCausalLM)
# https://huggingface.co/microsoft/biogpt 
# Boshra
# 02 September 2025

from transformers import BioGptTokenizer, BioGptForCausalLM, pipeline
import torch
import torch.nn.functional as F
from transformers import set_seed

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)

tokenizer = BioGptTokenizer.from_pretrained("microsoft/biogpt")
model = BioGptForCausalLM.from_pretrained("microsoft/biogpt")
model.to(device)

# -------------------------------
# Github Examples (modified):
# -------------------------------

# 1- Text generation
generator = pipeline('text-generation', model=model, tokenizer=tokenizer)
set_seed(42)
results = generator("COVID-19 is", max_length=20, num_return_sequences=5, do_sample=True)
print(results)

# 2- Here is how to use this model to get the features of a given text in PyTorch:
text = "Replace me by any text you'd like."
encoded_input = tokenizer(text, return_tensors='pt').to(device)
output = model(**encoded_input)

# 3- Beam-search decoding:
sentence = "COVID-19 is"
inputs = tokenizer(sentence, return_tensors="pt").to(device)

set_seed(42)
with torch.no_grad():
    beam_output = model.generate(**inputs,
                                min_length=100,
                                max_length=1024,
                                num_beams=5,
                                early_stopping=True
                                )
out = tokenizer.decode(beam_output[0], skip_special_tokens=True)
print(out)
'COVID-19 is a global pandemic caused by severe acute respiratory syndrome coronavirus 2 (SARS-CoV-2), the causative agent of coronavirus disease 2019 (COVID-19), which has spread to more than 200 countries and territories, including the United States (US), Canada, Australia, New Zealand, the United Kingdom (UK), and the United States of America (USA), as of March 11, 2020, with more than 800,000 confirmed cases and more than 800,000 deaths.'

# -------------------------------
# My exploration with BioGPT
# -------------------------------

# 1- Example of Text Generation (Pipeline)
def generation_with_pipeline(model, tokenizer):
    generator = pipeline("text-generation", model=model, tokenizer=tokenizer, device=0 if device.type=="cuda" else -1)
    set_seed(42)
    # sentence = "COVID-19 is"
    # sentence = "The patient was diagnosed with depression. The medicine he should take is"
    sentence = "The patient was diagnosed with depression. He should be treated with medication called"
    results = generator(sentence, max_length=40, num_return_sequences=3, do_sample=True)

    print("\nText Generation (pipeline):")
    for r in results:
        print(" ", r["generated_text"])


# 2- Manual next-token prediction
def manual_prediction(model, tokenizer):
    sentence = "The patient was prescribed medication named"
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    outputs = model(**inputs)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # Get last token logits
    next_token_logits = logits[:, -1, :]
    probs = F.softmax(next_token_logits, dim=-1)

    top_5 = torch.topk(probs, 5)
    print("\nNext token predictions (manual):")
    for i, idx in enumerate(top_5.indices[0]):
        token = tokenizer.decode([idx])
        score = top_5.values[0, i].item()
        print(f"  {token}: {score:.4f}")


# 3- Extract embeddings
def get_embeddings(model, tokenizer):
    sentence = "The patient was diagnosed with pneumonia."
    inputs = tokenizer(sentence, return_tensors="pt").to(device)

    with torch.no_grad():
        outputs = model.base_model(**inputs, output_hidden_states=True)

    hidden_states = outputs.hidden_states[-1]   # last layer hidden states
    cls_embedding = hidden_states[:, 0, :]      # [CLS]-like first token

    print("\nEmbeddings shape:", cls_embedding.shape)


# Run all
generation_with_pipeline(model, tokenizer)
manual_prediction(model, tokenizer)
get_embeddings(model, tokenizer)

print("\nDone.")









