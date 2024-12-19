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


def generate_class_words():
    """
    Generates extensive arrays of words for each class: DRINKTYPE, TOPPING, and STYLE.
    Returns a dictionary with these classes as keys.
    """
    class_words = {
        "DRINKTYPE": [
            "iced tea", "cola", "lemonade", "orange juice", "sparkling water", "coffee", "green tea", "latte",
            "cappuccino", "espresso", "milkshake", "smoothie", "mocha", "hot chocolate", "herbal tea", "black coffee",
            "americano", "frappuccino", "kombucha", "energy drink", "matcha latte", "cold brew", "ginger ale"
        ],
        "TOPPING": [
            "pepperoni", "mushrooms", "onions", "sausage", "bacon", "extra cheese", "black olives", "green peppers",
            "pineapple", "spinach", "anchovies", "jalapenos", "ham", "artichokes", "sun-dried tomatoes",
            "grilled chicken", "roasted garlic", "broccoli", "feta cheese", "arugula", "provolone", "salami",
            "prosciutto", "blue cheese", "chorizo", "truffle oil", "caramelized onions", "tofu", "basil", "eggplant"
        ],
        "STYLE": [
            "thin crust", "deep dish", "stuffed crust", "gluten-free", "hand-tossed", "New York style", "Sicilian style",
            "Neapolitan", "Detroit style", "Chicago style", "pan crust", "wood-fired", "crispy thin crust", "double crust",
            "cheese-stuffed crust", "cauliflower crust", "flatbread style", "cracker-thin crust", "artisan style",
            "vegan crust", "whole wheat crust"
        ]
    }
    return class_words


def augment_pizza_train(file_path, output_path, class_words, augmentation_probability=0.3):
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
augmented_output_path = "dataset/PIZZA_train_augmented.json"

# Execute the functions
clean_pizza_train_and_dev(train_path, dev_path, cleaned_output_path)
class_words = generate_class_words()
augment_pizza_train(cleaned_output_path, augmented_output_path, class_words)
