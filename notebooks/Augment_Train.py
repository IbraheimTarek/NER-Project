import json
import random
from copy import deepcopy
import re

def tokenize(s):
    # Extract tokens: parentheses or sequences of non-whitespace, non-parenthesis characters.
    tokens = re.findall(r'\(|\)|[^\s()]+', s)
    return tokens

def parse_tokens(tokens):
    # Parse tokens into a nested list structure
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

def clean_pizza_train(file_path, output_path):
    """
    Removes train.exr and train.top-decoupled fields from the JSON file.
    """
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            src_text = record["train.SRC"]
            src_top = record["train.TOP"]
            outfile.write(json.dumps({"src": src_text, "top": src_top}) + "\n")


def generate_class_words():
    """
    Generates arrays of words for each class: DRINKTYPE, TOPPING, and STYLE.
    Returns a dictionary with these classes as keys.
    """
    class_words = {
        "DRINKTYPE": [
            "iced tea", "cola", "lemonade", "orange juice", "sparkling water", "coffee", "green tea", "latte", "cappuccino",
            "espresso", "milkshake", "smoothie", "mocha", "hot chocolate"
        ],
        "TOPPING": [
            "pepperoni", "mushrooms", "onions", "sausage", "bacon", "extra cheese", "black olives", "green peppers",
            "pineapple", "spinach", "anchovies", "jalapenos", "ham", "artichokes", "sun-dried tomatoes"
        ],
        "STYLE": [
            "thin crust", "deep dish", "stuffed crust", "gluten-free", "hand-tossed", "New York style", "Sicilian style",
            "Neapolitan", "Detroit style", "Chicago style"
        ]
    }
    return class_words

def augment_pizza_train(file_path, output_path, class_words, augmentation_probability=0.3):
    """
    Augments the PIZZA_train dataset by replacing words in specific classes with new ones.
    
    Args:
        file_path (str): Path to the original dataset.
        output_path (str): Path to save the augmented dataset.
        class_words (dict): Words for each class.
        augmentation_probability (float): Probability of replacing a word in the specified classes.
    """
    with open(file_path, 'r') as file:
        data = json.load(file)

    augmented_data = deepcopy(data)

    for record in data:
        src = record["src"]
        top = record["top"]

        # Tokenize the `top` string and parse the structure
        top_tokens = tokenize(top)
        top_structure = parse_tokens(top_tokens)

        # Identify and replace class-specific words
        def replace_tokens(structure):
            if not isinstance(structure, list):
                return structure

            if len(structure) > 0 and structure[0] in class_words:
                class_type = structure[0]
                if random.random() < augmentation_probability:
                    # Replace tokens with random choice from class_words[class_type]
                    replacement = random.choice(class_words[class_type]).split()
                    return [class_type] + replacement

            return [replace_tokens(sub) if isinstance(sub, list) else sub for sub in structure]

        # Apply replacements to the top structure
        new_top_structure = replace_tokens(top_structure)

        # Rebuild `top` string from the modified structure
        def rebuild_structure(structure):
            if not isinstance(structure, list):
                return structure

            result = "(" + " ".join(rebuild_structure(sub) for sub in structure) + ")"
            return result

        new_top = rebuild_structure(new_top_structure)

        # Update `src` by replacing corresponding words
        new_src = src
        for original, replacements in zip(top_structure, new_top_structure):
            if isinstance(original, list) and isinstance(replacements, list):
                if original[0] in class_words:
                    original_text = " ".join(original[1:])
                    replacement_text = " ".join(replacements[1:])
                    new_src = new_src.replace(original_text, replacement_text)

        # Append the new record
        augmented_data.append({"src": new_src, "top": new_top})

    with open(output_path, 'w') as file:
        json.dump(augmented_data, file, indent=4)
        
    with open(file_path, 'r') as infile, open(output_path, 'w') as outfile:
        for line in infile:
            record = json.loads(line)
            src_text = record["train.SRC"]
            src_top = record["train.TOP"]
            
            top_tokens = tokenize(src_top)
            top_structure = parse_tokens(top_tokens)
            new_top_structure = replace_tokens(top_structure)
            new_top = rebuild_structure(new_top_structure)
            
            new_src = src_text
            for original, replacements in zip(top_structure, new_top_structure):
                if isinstance(original, list) and isinstance(replacements, list):
                    if original[0] in class_words:
                        original_text = " ".join(original[1:])
                        replacement_text = " ".join(replacements[1:])
                        new_src = new_src.replace(original_text, replacement_text)
            outfile.write(json.dumps({"src": new_src, "top": new_top}) + "\n")
        
    

# Paths
input_path = "dataset/PIZZA_train.json"
cleaned_output_path = "dataset/PIZZA_train_cleaned.json"
augmented_output_path = "dataset/PIZZA_train_augmented.json"

# Execute the functions
clean_pizza_train(input_path, cleaned_output_path)
class_words = generate_class_words()
augment_pizza_train(cleaned_output_path, augmented_output_path, class_words)
