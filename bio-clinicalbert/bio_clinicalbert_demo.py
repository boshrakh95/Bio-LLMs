from transformers import AutoTokenizer, AutoModelForMaskedLM, pipeline

# Load Bio_ClinicalBERT
model_name = "emilyalsentzer/Bio_ClinicalBERT"

print(f"Loading model: {model_name}")
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForMaskedLM.from_pretrained(model_name)

# Create a fill-mask pipeline
fill_mask = pipeline("fill-mask", model=model, tokenizer=tokenizer)

# Example sentence with [MASK] token
sentence = "The patient was diagnosed with [MASK] cancer."

print("\nInput:", sentence)
results = fill_mask(sentence)

print("\nPredictions:")
for r in results:
    print(f"  {r['sequence']} (score: {r['score']:.4f})")
