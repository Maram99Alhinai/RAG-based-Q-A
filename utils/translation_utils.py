import ctranslate2
import transformers

def translate_text(text, source_lang, target_lang, model_path):
    """Translates text using CTranslate2."""
    translator = ctranslate2.Translator(model_path)
    tokenizer = transformers.AutoTokenizer.from_pretrained(model_path)
    tokens = tokenizer.convert_ids_to_tokens(tokenizer.encode(text))
    results = translator.translate_batch([tokens], target_prefix=[[target_lang]])
    translated_tokens = results[0].hypotheses[0]
    translated_text = tokenizer.decode(tokenizer.convert_tokens_to_ids(translated_tokens))
    return translated_text

def summarize_text(llm, text):
    """Summarizes text using the LLM."""
    prompt = f"Summarize the following text:\n{text}"
    summary = llm(prompt)
    return summary