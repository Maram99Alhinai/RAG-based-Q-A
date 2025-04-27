from transformers import pipeline

def translate(text, model_path, source_lang, target_lang):
    translator = pipeline("translation", model=model_path, src_lang=source_lang, tgt_lang=target_lang)
    return translator(text)[0]['translation_text']

print(translate('Changed to the new model', "models\m2m100_418M", source_lang = 'en', target_lang = 'ar'))
