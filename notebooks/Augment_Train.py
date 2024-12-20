import json
import random
import re


def tokenize(s):
    tokens = re.findall(r'\(|\)|[^\s()]+', s)
    return tokens


def parse_tokens(tokens):
    stack = []
    current_list = []
    for token in tokens:
        if token == '(':
            stack.append(current_list)
            current_list = []
        elif token == ')':
            finished = current_list
            current_list = stack.pop()
            current_list.append(finished)
        else:
            current_list.append(token)
    return current_list


def clean_pizza_train_and_dev(train_path, dev_path, output_path):
    """
    Cleans and merges PIZZA_train and PIZZA_dev datasets, keeping only `src` and `top`.
    """
    with open(output_path, 'w') as outfile:
        # Process PIZZA_train
        with open(train_path, 'r') as train_file:
            for line in train_file:
                record = json.loads(line)
                src_text = record["train.SRC"]
                src_top = record["train.TOP"]
                outfile.write(json.dumps({"train.SRC": src_text, "train.TOP": src_top}) + "\n")
        
        # Process PIZZA_dev
        with open(dev_path, 'r') as dev_file:
            for line in dev_file:
                record = json.loads(line)
                src_text = record["dev.SRC"]
                src_top = record["dev.TOP"]
                outfile.write(json.dumps({"train.SRC": src_text, "train.TOP": src_top}) + "\n")
                
        # process PIZZA.test
        with open("dataset/PIZZA_test.json", 'r') as test_file:
            for line in test_file:
                record = json.loads(line)
                src_text = record["test.SRC"]
                src_top = record["test.TOP"]
                outfile.write(json.dumps({"train.SRC": src_text, "train.TOP": src_top}) + "\n")


def generate_class_words():
    """
    Generates extensive arrays of words for each class: DRINKTYPE, TOPPING, and STYLE.
    Returns a dictionary with these classes as keys.
    """
    class_words = {
    "DRINKTYPE": [
        "iced tea", "iced cofee", "cola", "kola", "lemonade", "lemonaid", "orange juice", "orrange juice",
        "sparkling water", "sparklin water", "coffee", "coffe", "green tea", "greeen tea", "latte", "lattee",
        "cappuccino", "capuccino", "expresso", "espresso", "milkshake", "mylkshake", "smoothie", "smoothe",
        "mocha", "mochaa", "hot chocolate", "herbal tea", "herbal tee", "black coffee", "blak coffee",
        "americano", "amerikano", "frappuccino", "frapucino", "kombucha", "kombutcha", "energy drink",
        "enrgy drink", "matcha latte", "matca latte", "cold brew", "cold breww", "ginger ale", "ginger ail",
        "bubble tea", "buble tea", "chai latte", "chia latte", "turmeric latte", "turmeric latee"
    ],
    "TOPPING": [
        "pepperoni", "peperoni", "mushrooms", "mushroms", "onions", "onionz", "sausage", "sausge",
        "bacon", "bacn", "extra cheese", "extra cheez", "black olives", "blak olives", "green peppers", 
        "grn peppers", "pineapple", "pinaple", "spinach", "spinich", "anchovies", "anchovis", "jalapenos",
        "jalapinos", "ham", "grilled chicken", "grilled chiken", "roasted garlic", "roastd garlic",
        "broccoli", "brocli", "feta cheese", "fetta cheese", "arugula", "arrugula", "provolone", "provlone",
        "salami", "sallami", "prosciutto", "proscutto", "blue cheese", "blu cheese", "chorizo", "choriso",
        "truffle oil", "truffl oil", "caramelized onions", "caramlized onions", "tofu", "basil", "bazil",
        "eggplant", "egplant", "artichokes", "artchokes", "sun-dried tomatoes", "sun dried tomatos"
    ],
    "STYLE": [
        "thin crust", "thn crust", "deep dish", "deap dish", "stuffed crust", "stuffd crust", "gluten-free",
        "gluten freee", "hand-tossed", "hand tossed", "New York style", "NY style", "Sicilian style",
        "Silician style", "Neapolitan", "Napolitan", "Detroit style", "Detriot style", "Chicago style",
        "pan crust", "pancrust", "wood-fired", "wood fired", "crispy thin crust", "cripsy thin crust",
        "double crust", "dubble crust", "cheese-stuffed crust", "cheesestuffed crust", "cauliflower crust",
        "cauli flower crust", "flatbread style", "flat bread style", "cracker-thin crust", "artisan style",
        "artesan style", "vegan crust", "vgn crust", "whole wheat crust", "whole wheet crust"
    ]
    }

    return class_words


def augment_pizza_train(file_path, output_path, class_words, augmentation_probability=0.5):
    """
    Augments the PIZZA_train dataset by replacing words in specific classes with new ones.
    """
    def replace_tokens(structure, src_text, is_changed):
        """
        Replace tokens in the given structure with augmented words and update src_text.
        Tracks whether any replacements have been made using is_changed.
        """
        if not isinstance(structure, list):
            return structure, src_text, is_changed

        # Check if the first token is a class type
        if len(structure) > 0 and isinstance(structure[0], str) and structure[0] in class_words:
            class_type = structure[0]
            if random.random() < augmentation_probability:
                # Replace tokens with random choice from class_words[class_type]
                replacement = random.choice(class_words[class_type]).split()
                original_text = " ".join(structure[1:])
                replacement_text = " ".join(replacement)
                # Update src_text with the replacement
                src_text = src_text.replace(original_text, replacement_text)
                is_changed = True
                return [class_type] + replacement, src_text, is_changed

        # Recursively process nested structures
        new_structure = []
        for sub in structure:
            replaced, src_text, is_changed = replace_tokens(sub, src_text, is_changed)
            new_structure.append(replaced)

        return new_structure, src_text, is_changed

    def rebuild_structure(structure):
        """
        Rebuilds the nested structure into a string format.
        """
        if not isinstance(structure, list):
            return structure
        result = "(" + " ".join(rebuild_structure(sub) for sub in structure) + ")"
        return result

    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            src_text = record["train.SRC"]
            src_top = record["train.TOP"]

            # Tokenize and parse the TOP structure
            top_tokens = tokenize(src_top)
            top_structure = parse_tokens(top_tokens)

            # Replace tokens in the structure and update SRC
            is_changed = False
            new_top_structure, new_src, is_changed = replace_tokens(top_structure, src_text, is_changed)

            # Rebuild the TOP structure into a string
            new_top = rebuild_structure(new_top_structure)

            # Write the augmented record only if it has changes
            if is_changed:
                outfile.write(json.dumps({"train.SRC": new_src, "train.TOP": new_top}) + "\n")

            
    # Append original cleaned records to the output file
    with open(file_path, 'r') as infile, open(output_path, 'a') as outfile:
        # read 2 lines of every 10 lines
        count = 0
        for line in infile:
            if count % 10 == 0 or count % 10 == 1:
                outfile.write(line)
            count += 1

# Paths
train_path = "dataset/PIZZA_train.json"
dev_path = "dataset/PIZZA_dev.json"
cleaned_output_path = "dataset/PIZZA_train_cleaned.json"
augmented_output_path = "dataset3/PIZZA_train.json"

# Execute the functions
clean_pizza_train_and_dev(train_path, dev_path, cleaned_output_path)
class_words = generate_class_words()
augment_pizza_train(cleaned_output_path, augmented_output_path, class_words)
