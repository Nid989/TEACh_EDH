from transformers import AutoModelForSequenceClassification

model_name = "llama"
num_labels = 2 # replace with the actual number of labels in your classification task

model = AutoModelForSequenceClassification.from_pretrained(model_name, num_labels=num_labels)