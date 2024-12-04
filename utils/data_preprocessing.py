import re
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
import nltk
from nltk.corpus import stopwords
 
nltk.download('stopwords')
lemmatizer = WordNetLemmatizer()


def clean_text(text):
    """
    Remove special characters and unnecessary symbols from text.
    """
    #stop_words = set(stopwords.words('english')) # takes much time
    stop_words = [
    "a", "an", "the", "and", "or", "but", "if", "on", "in", "with", "without", "at", 
    "by", "from", "to", "of", "for", "this", "that", "those", "these", 
    "can", "could", "would", "should", "will", "might", "may", "i", "you", 
    "we", "he", "she", "it", "they", "is", "are", "was", "were", "be", 
    "been", "have", "has", "had", "please"
    ]
    custom_remove = [
    r"extra\s", 
    r"please",
    r"thank\s?you", 
    r"no\s",     # Remove negations if context-independent
    r"lot\s?of", # Remove "lot of"
    r"kindly", 
    r"just", 
    r"really",
    r"actually"
    ]

    # Remove special characters
    text = re.sub(r"[^\w\s]", " ", text)  # Remove punctuation and special characters
    
    # Remove extra whitespace
    text = re.sub(r"\s+", " ", text).strip()
    
    # Remove stopwords
    if stop_words:
        text = " ".join([word for word in text.split() if word.lower() not in stop_words])
    
    # Remove custom characters or substrings
    if custom_remove:
        for pattern in custom_remove:
            text = re.sub(pattern, "", text)
    return text

def tokenize_text(text):
    """
    Tokenize the text into words.
    """
    return word_tokenize(text)

def lemmatize_tokens(tokens):
    """
    Lemmatize each token in a list.
    """
    return [lemmatizer.lemmatize(token) for token in tokens]

def preprocess_text(text):
    """
    Full preprocessing pipeline.
    """
    text = clean_text(text)
    tokens = tokenize_text(text)
    tokens = lemmatize_tokens(tokens)
    return tokens


