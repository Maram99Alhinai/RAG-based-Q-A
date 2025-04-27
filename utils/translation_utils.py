from transformers import pipeline



def translate(text, model_path, source_lang, target_lang):
    translator = pipeline("translation", model=model_path, src_lang=source_lang, tgt_lang=target_lang)
    return translator(text)[0]['translation_text']
