# Clinical Camel: https://huggingface.co/wanglab/ClinicalCamel-70B
# Boshra
# 01 September 2025

# IMPORTANT NOTE:
# I couldn't run it because the model is huge and loading it needs downloading multiple 10GB files.
# The model is ~150 GB in total. Cannot avoid downloading all shards if you want to use the full 70B model.

from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
import torch

# -------------------------------
# Device setup
# -------------------------------
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

# -------------------------------
# Load tokenizer and model
# -------------------------------
model_name = "wanglab/ClinicalCamel-70B"
print(f"Loading model: {model_name}")

tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(
    model_name,
    torch_dtype=dtype,
    device_map="auto",          # automatically offloads to CPU/GPU
    offload_folder="offload"    # optional, large models benefit from offloading
)

# -------------------------------
# 1. Text generation using pipeline
# -------------------------------
gen_pipe = pipeline(
    "text-generation",
    model=model,
    tokenizer=tokenizer,
)

prompt = "A patient presents with chest pain and shortness of breath. Likely diagnosis:"
print("\nPrompt:", prompt)

generated = gen_pipe(
    prompt,
    max_new_tokens=5,   # adjust as needed
    do_sample=True,
    temperature=0.7,
    top_p=0.9
)
print("\nGenerated Text:\n", generated[0]["generated_text"])

# -------------------------------
# 2. Manual generation (tensor-level)
# -------------------------------
inputs = tokenizer(prompt, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.generate(
        **inputs,
        max_new_tokens=5,
        do_sample=True,
        temperature=0.7,
        top_p=0.9
    )

manual_generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
print("\nManual Generation:\n", manual_generated_text)

# -------------------------------
# 3. Get embeddings from the model
# -------------------------------
# Clinical Camel is decoder-only, last hidden states can be used as token embeddings
text = "The patient was prescribed 10mg of Lisinopril."
inputs = tokenizer(text, return_tensors="pt").to(device)
with torch.no_grad():
    outputs = model.base_model(**inputs)

last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
cls_embedding = last_hidden_state[:, 0, :]     # first token embedding
print("\nCLS-like embedding shape:", cls_embedding.shape)
