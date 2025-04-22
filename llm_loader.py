from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
import os



#Translate model 
model_name = "Helsinki-NLP/opus-mt-en-fr"
save_directory = "models"

# Create the directory if it doesn't exist
os.makedirs(save_directory, exist_ok=True)

# Download tokenizer and model
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForSeq2SeqLM.from_pretrained(model_name)

# Save them to the specified directory
tokenizer.save_pretrained(os.path.join(save_directory, "opus-mt-en-fr"))
model.save_pretrained(os.path.join(save_directory, "opus-mt-en-fr"))

print(f"OPUS-MT en-fr model and tokenizer downloaded to: {os.path.join(save_directory, 'opus-mt-en-fr')}")



