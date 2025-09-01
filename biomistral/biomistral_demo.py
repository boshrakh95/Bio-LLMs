# BioMistral: https://huggingface.co/BioMistral/BioMistral-7B 
# Boshra
# 29 August 2025

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# Device setup
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)
print("Using device:", device)

dtype = (
    torch.bfloat16 if device.type == "cuda"
    else torch.float16 if device.type == "mps"
    else torch.float32
)

# Load tokenizer and model
model_name = "BioMistral/BioMistral-7B"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_name, torch_dtype=dtype, device_map="auto", offload_folder="offload")  # offload_folder takes 9-10GB
# model.to(device)

# -------------------------------
# 1. Text generation using pipeline
# -------------------------------
gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
    # device=-1,
)

prompt = "A patient presents with severe headache and nausea. Likely diagnosis:"
print("\nPrompt:", prompt)

generated = gen_pipe(prompt, max_new_tokens=20, do_sample=False, temperature=0.7) # reduced max_new_tokens to take less time
print("\nGenerated Text:\n", generated[0]["generated_text"])

# -------------------------------
# 2. Manual generation (tensor-level)
# -------------------------------
inputs = tokenizer(prompt, return_tensors="pt")
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=20,
        do_sample=False,
        temperature=0.7,
        top_p=0.9
    )

manual_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nManual Generation:\n", manual_generated_text)

# -------------------------------
# 3. Get embeddings from the model
# -------------------------------
# BioMistral is decoder-only, so we can use last hidden states as token embeddings
text = "The patient was prescribed 5mg of Coumadin."
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.base_model(**inputs)

last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
cls_embedding = last_hidden_state[:, 0, :]  # first token embedding
print("\nCLS-like embedding shape:", cls_embedding.shape)
