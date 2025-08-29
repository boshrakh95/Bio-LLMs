from transformers import AutoModel, AutoTokenizer, AutoModelForMaskedLM, pipeline
import torch
import torch.nn.functional as F

def prediction_with_pipeline(full_model, tokenizer):

    # Create a fill-mask pipeline
    fill_mask = pipeline("fill-mask", model=full_model, tokenizer=tokenizer)

    # Example sentence with [MASK] token
    sentence = "The patient was diagnosed with [MASK] cancer."

    print("\nInput:", sentence)
    results = fill_mask(sentence)

    print("\nPredictions:")
    for r in results:
        print(f"  {r['sequence']} (score: {r['score']:.4f})")


def prediction_manual(full_model, tokenizer):

    # Example sentence with [MASK]
    sentence = "The patient was diagnosed with [MASK] cancer."
    inputs = tokenizer(sentence, return_tensors="pt")
    inputs = {k: v.to(device) for k, v in inputs.items()}

    # Find the [MASK] token index
    mask_token_index = torch.where(inputs["input_ids"] == tokenizer.mask_token_id)[1]

    # Forward pass to get logits
    outputs = full_model(**inputs)
    logits = outputs.logits  # shape: [1, seq_len, vocab_size]

    # Get logits for the masked position
    mask_logits = logits[0, mask_token_index, :]  # shape: [1, vocab_size]

    # Convert logits to probabilities
    probs = F.softmax(mask_logits, dim=-1)

    # Get top 5 predictions
    top_5 = torch.topk(probs, 5, dim=-1)

    for i, idx in enumerate(top_5.indices[0]):
        token = tokenizer.decode([idx])
        score = top_5.values[0, i].item()
        print(f"{token}: {score:.4f}")


# Load Bio_ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"
tokenizer = AutoTokenizer.from_pretrained(model_name)

print(f"Loading model: {model_name}")

device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
    )

# 1- Example of prediction 

# Load model with Masked Language Modeling head on top
full_model = AutoModelForMaskedLM.from_pretrained(model_name, device_map=None,       # MPS doesn’t support `device_map="auto"`
    torch_dtype=torch.float32)
full_model.to(device)

prediction_with_pipeline(full_model, tokenizer) # example of using pipeline for easy inference

prediction_manual(full_model, tokenizer) # example of manual inference


# 2- Example of Getting Embeddings (Encode text)

# Load model without any task-specific head.
base_model = AutoModel.from_pretrained(model_name)

text = "The patient was prescribed 5mg of Coumadin."
inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
outputs = base_model(**inputs)

# Extract embeddings
last_hidden_state = outputs.last_hidden_state  # shape: (1, seq_len, hidden_dim)
cls_embedding = last_hidden_state[:, 0, :]     # [CLS] token embedding
