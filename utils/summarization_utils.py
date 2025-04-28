from transformers import pipeline

def summarize_text(text, model_path='models\\bart-large-cnn', max_length=100, min_length=30):

    try:
        summarizer = pipeline("summarization", model=model_path)
        summary = summarizer(text, max_length=max_length, min_length=min_length, do_sample=False)[0]['summary_text']
        return summary
    except Exception as e:
        return f"An error occurred: {e}"
