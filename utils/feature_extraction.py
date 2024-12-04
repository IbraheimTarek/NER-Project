from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer
import numpy as np
import gensim.downloader as api

# Load pre-trained embeddings (e.g., GloVe)
glove_vectors = api.load("glove-wiki-gigaword-100")  # 100-dimension GloVe

def bag_of_words(corpus):
    """
    Extract Bag of Words features from a text corpus.
    """
    vectorizer = CountVectorizer()
    bow_features = vectorizer.fit_transform(corpus)
    return bow_features, vectorizer

def tfidf_features(corpus):
    """
    Extract TF-IDF features from a text corpus.
    """
    vectorizer = TfidfVectorizer()
    tfidf_features = vectorizer.fit_transform(corpus)
    return tfidf_features, vectorizer

def get_word_embeddings(tokens):
    """
    Get average word embeddings for a list of tokens using pre-trained GloVe.
    """
    embeddings = []
    for token in tokens:
        if token in glove_vectors:
            embeddings.append(glove_vectors[token])
    if embeddings:
        return np.mean(embeddings, axis=0)
    else:
        return np.zeros(glove_vectors.vector_size)  # Return a zero vector if no embeddings are found

def extract_embeddings(corpus):
    """
    Extract word embeddings for a corpus.
    """
    return np.array([get_word_embeddings(tokens) for tokens in corpus])
