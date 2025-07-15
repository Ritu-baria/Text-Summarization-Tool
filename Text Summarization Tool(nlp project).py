import nltk
import spacy
from heapq import nlargest

# Download NLTK tokenizer
nltk.download('punkt')

# Load spacy English model
nlp = spacy.load("en_core_web_sm")

# Input long text
text = """
Artificial Intelligence is transforming every industry. From healthcare to finance, the impact of AI can be seen in improving accuracy, speed, and personalization.
Text summarization is one key area of NLP, used in news aggregation, document classification, and reducing information overload.
This tool is built with Spacy and NLTK for extractive summarization by identifying and ranking the most important sentences.
"""

# Preprocess text with Spacy
doc = nlp(text)
stopwords = spacy.lang.en.stop_words.STOP_WORDS
word_freq = {}

# Calculate word frequencies excluding stopwords
for token in doc:
    if token.text.lower() not in stopwords and token.text.isalpha():
        word = token.text.lower()
        word_freq[word] = word_freq.get(word, 0) + 1

# Normalize word frequencies
max_freq = max(word_freq.values())
for word in word_freq:
    word_freq[word] /= max_freq

# Sentence scoring
sentence_scores = {}
sentences = nltk.sent_tokenize(text)

for sent in sentences:
    sent_doc = nlp(sent)
    for word in sent_doc:
        if word.text.lower() in word_freq:
            sentence_scores[sent] = sentence_scores.get(sent, 0) + word_freq[word.text.lower()]

# Top N sentences for summary
summary_sentences = nlargest(2, sentence_scores, key=sentence_scores.get)
summary = ' '.join(summary_sentences)

# Output
print("📝 Summary:\n", summary)
