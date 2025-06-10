# article_summarizer.py

from transformers import pipeline

def summarize_article(text):
    """
    Summarizes lengthy article text using a pre-trained NLP model.

    Parameters:
    - text (str): The full input article.

    Returns:
    - summary (str): A concise summary of the input text.
    """
    summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
    summary_output = summarizer(text, max_length=130, min_length=30, do_sample=False)
    return summary_output[0]['summary_text']

if __name__ == "__main__":
    # 📝 Sample article input (You can replace this with any long text)
    article_text = """
    Artificial Intelligence (AI) is revolutionizing industries across the globe. From automating routine tasks to 
    providing intelligent recommendations, AI is making human lives easier and businesses more efficient. Among 
    its many branches, Natural Language Processing (NLP) is particularly noteworthy for its ability to enable 
    computers to understand and generate human language. This capability has led to the development of chatbots, 
    translation tools, and powerful summarization systems that can digest large volumes of text into short summaries. 
    These tools save time, improve accessibility, and are increasingly used in education, journalism, and law.
    """

    print("\n📄 Original Article:\n")
    print(article_text)

    print("\n🔍 Summarizing...\n")
    summary = summarize_article(article_text)

    print("📝 Summary:\n")
    print(summary)
