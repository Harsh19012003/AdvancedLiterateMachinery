from transformers import AutoTokenizer
tokenizer = "google-bert/bert-base-uncased"
tokenizer = AutoTokenizer.from_pretrained(tokenizer)
print(tokenizer)
