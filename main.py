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

{"train.SRC": "large pie with green pepper and with extra peperonni", "train.EXR": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE LARGE ) (TOPPING GREEN_PEPPERS ) (COMPLEX_TOPPING (QUANTITY EXTRA ) (TOPPING PEPPERONI ) ) ) )", "train.TOP": "(ORDER (PIZZAORDER (SIZE large ) pie with (TOPPING green pepper ) and with (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) ) )", "train.TOP-DECOUPLED": "(ORDER (PIZZAORDER (SIZE large ) (TOPPING green pepper ) (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING peperonni ) ) ) )"}
{"train.SRC": "i'd like a large vegetarian pizza", "train.EXR": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE LARGE ) (STYLE VEGETARIAN ) ) )", }
{"train.SRC": "party size stuffed crust pie with american cheese and with mushroom", "train.EXR": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE PARTY_SIZE ) (STYLE STUFFED_CRUST ) (TOPPING AMERICAN_CHEESE ) (TOPPING MUSHROOMS ) ) )", "train.TOP": "(ORDER (PIZZAORDER (SIZE party size ) (STYLE stuffed crust ) pie with (TOPPING american cheese ) and with (TOPPING mushroom ) ) )", "train.TOP-DECOUPLED": "(ORDER (PIZZAORDER (SIZE party size ) (STYLE stuffed crust ) (TOPPING american cheese ) (TOPPING mushroom ) ) )"}
{"train.SRC": "can i have one personal sized artichoke", "train.EXR": "(ORDER (PIZZAORDER (NUMBER 1 ) (SIZE PERSONAL_SIZE ) (TOPPING ARTICHOKES ) ) )", "train.TOP": "(ORDER can i have (PIZZAORDER (NUMBER one ) (SIZE personal sized ) (TOPPING artichoke ) ) )", "train.TOP-DECOUPLED": "(ORDER (PIZZAORDER (NUMBER one ) (SIZE personal sized ) (TOPPING artichoke ) ) )"}
{"train.SRC": "pie with banana pepper and peppperonis and extra low fat cheese", "train.EXR": "(ORDER (PIZZAORDER (NUMBER 1 ) (TOPPING BANANA_PEPPERS ) (TOPPING PEPPERONI ) (COMPLEX_TOPPING (QUANTITY EXTRA ) (TOPPING LOW_FAT_CHEESE ) ) ) )", "train.TOP": "(ORDER (PIZZAORDER pie with (TOPPING banana pepper ) and (TOPPING peppperonis ) and (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING low fat cheese ) ) ) )", "train.TOP-DECOUPLED": "(ORDER (PIZZAORDER (TOPPING banana pepper ) (TOPPING peppperonis ) (COMPLEX_TOPPING (QUANTITY extra ) (TOPPING low fat cheese ) ) ) )"}


