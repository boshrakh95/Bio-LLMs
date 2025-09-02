# https://huggingface.co/michiyasunaga/BioLinkBERT-base
# For fine-tuning code, refer to: https://github.com/michiyasunaga/LinkBERT 
# Boshra
# 01 September 2025

# IMPORTANT NOTE:
# Their huggingface says the model can be used by fine-tuning on a downstream task, such as question answering, sequence classification, and token classification. 
# You can also use the raw model for feature extraction (i.e. obtaining embeddings for input text).
# So the model does not perform well on QA and other tasks without fine-tuning (on their github).
# Running the following code for different tasks will give very bad results. Only use the embedding extraction part if needed.

from transformers import (
    AutoModel, AutoTokenizer,
    AutoModelForQuestionAnswering,
    AutoModelForMultipleChoice,
    AutoModelForSequenceClassification,
    pipeline
)
import torch
import torch.nn.functional as F


# ======================
# Helper: Select Device
# ======================
device = (
    torch.device("cuda") if torch.cuda.is_available()
    else torch.device("mps") if torch.backends.mps.is_available()
    else torch.device("cpu")
)


# ======================
# 1- Question Answering
# ======================
def qa_with_pipeline(model_name, question, context):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    qa_pipe = pipeline("question-answering", model=model, tokenizer=tokenizer, device=-1)
    result = qa_pipe(question=question, context=context)

    print("\n[QA Prediction with pipeline]")
    print(f"Question: {question}")
    print(f"Answer: {result['answer']} (score: {result['score']:.4f})")


def qa_manual(model_name, question, context):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForQuestionAnswering.from_pretrained(model_name).to(device)

    inputs = tokenizer(question, context, return_tensors="pt").to(device)
    outputs = model(**inputs)

    start_logits, end_logits = outputs.start_logits, outputs.end_logits
    start_idx = torch.argmax(start_logits)
    end_idx = torch.argmax(end_logits) + 1

    answer = tokenizer.convert_tokens_to_string(
        tokenizer.convert_ids_to_tokens(inputs["input_ids"][0][start_idx:end_idx])
    )

    print("\n[QA Prediction manual]")
    print(f"Answer: {answer}")


# ======================
# 2- Multiple Choice
# ======================
def multiple_choice_manual(model_name, question, options):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForMultipleChoice.from_pretrained(model_name).to(device)

    # Format inputs: replicate question for each option
    inputs = tokenizer(
        [question] * len(options), options,
        return_tensors="pt", padding=True, truncation=True
    ).to(device)

    # Reshape for (batch_size=1, num_choices, seq_len)
    inputs = {k: v.unsqueeze(0) for k, v in inputs.items()}

    outputs = model(**inputs)
    logits = outputs.logits  # shape: (1, num_choices)
    probs = torch.softmax(logits, dim=-1)

    best_idx = torch.argmax(probs, dim=-1).item()

    print("\n[Multiple Choice Prediction]")
    for i, (opt, score) in enumerate(zip(options, probs[0])):
        print(f"  {opt} (score: {score:.4f})")
    print(f"Predicted answer: {options[best_idx]}")


# ======================
# 3- Sequence Classification
# ======================
def seq_classification_manual(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=2).to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = model(**inputs)

    logits = outputs.logits
    probs = torch.softmax(logits, dim=-1)

    print("\n[Sequence Classification Prediction]")
    print(f"Text: {text}")
    print(f"Probabilities: {probs.detach().cpu().numpy()}")


# ======================
# 4- Embeddings Extraction
# ======================
def get_embeddings(model_name, text):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    base_model = AutoModel.from_pretrained(model_name).to(device)

    inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True).to(device)
    outputs = base_model(**inputs)

    # Option A: [CLS] token hidden state
    cls_embedding = outputs.last_hidden_state[:, 0, :]

    # Option B: pooled output (BioLinkBERT/ BERT-style models have it)
    if hasattr(outputs, "pooler_output") and outputs.pooler_output is not None:
        pooled_embedding = outputs.pooler_output
    else:
        pooled_embedding = cls_embedding  # fallback

    print("\n[Embeddings Extraction]")
    print(f"CLS embedding shape: {cls_embedding.shape}")
    print(f"Pooled embedding shape: {pooled_embedding.shape}")

    return pooled_embedding


# ======================
# Run Examples
# ======================
if __name__ == "__main__":
    model_name = "michiyasunaga/BioLinkBERT-base"

    # QA
    context = "Sunitinib is a tyrosine kinase inhibitor used in cancer treatment."
    question = "What does Sunitinib inhibit?"
    qa_with_pipeline(model_name, question, context)
    qa_manual(model_name, question, context)

    # Multiple Choice
    mc_question = "What type of drug is Sunitinib?"
    options = ["Antibiotic", "Tyrosine kinase inhibitor", "Beta-blocker"]
    multiple_choice_manual(model_name, mc_question, options) # output: antibiotic, which is wrong

    # Sequence Classification
    seq_classification_manual(model_name, "The patient was diagnosed with diabetes.")

    # Embeddings
    get_embeddings(model_name, "The patient was prescribed 5mg of Coumadin.")
