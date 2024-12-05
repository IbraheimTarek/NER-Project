from utils import preprocess_text, bag_of_words, tfidf_features, extract_embeddings

# Example text corpus
corpus = [
    "i want to order two medium pizzas with sausage and black olives and two medium pizzas with pepperoni and extra cheese and three large pizzas with pepperoni and sausage",
    "can i have one high rise dough pie with american cheese and a lot of meatball",
    "i'd like one party - size pie with no cheese"
]
# Preprocess the text testing
preprocessed_corpus = [" ".join(preprocess_text(text)) for text in corpus]
print(preprocessed_corpus)

# embedding testing
#Bag of Words
bow_features, bow_vectorizer = bag_of_words(preprocessed_corpus)
print("Bag of Words Features:\n", bow_features.toarray())

# TF-IDF
tfidf_features, tfidf_vectorizer = tfidf_features(preprocessed_corpus)
print("TF-IDF Features:\n", tfidf_features.toarray())

# Word Embeddings
embeddings = extract_embeddings([text.split() for text in preprocessed_corpus])
print("Word Embeddings:\n", embeddings)

