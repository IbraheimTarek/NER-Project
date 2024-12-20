import streamlit as st
import json
import torch
import torch.nn as nn
import torch.nn.functional as F
import re
import json
import re

def tokenize_top(s):
    # Extract tokens: parentheses or sequences of non-whitespace, non-parenthesis characters.
    tokens = re.findall(r'\(|\)|[^\s()]+', s)
    return tokens

def tokenize_input(s):
    return s.split()

def tokens_to_ints(tokens, vocab):
    # map the tokens to integers from vocab
    # if the token is not in the vocab, use the index of the unknown token
    return [vocab.get(token, vocab['<UNK>']) for token in tokens]

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

def normalize_structure(tree):
    if not isinstance(tree, list):
        return None

    def is_key(token):
        return token in [
            "ORDER", "PIZZAORDER", "DRINKORDER", "NUMBER", "SIZE", "STYLE", "TOPPING",
            "COMPLEX_TOPPING", "QUANTITY", "VOLUME", "DRINKTYPE", "CONTAINERTYPE", "NOT"
        ]

    # Clean the list by keeping sublists and tokens as-is for further analysis
    cleaned = []
    for el in tree:
        cleaned.append(el)

    if len(cleaned) > 0 and isinstance(cleaned[0], str) and is_key(cleaned[0]):
        key = cleaned[0]
        if key == "ORDER":
            pizzaorders = []
            drinkorders = []
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    if "PIZZAORDER" in node:
                        if isinstance(node["PIZZAORDER"], list):
                            pizzaorders.extend(node["PIZZAORDER"])
                        else:
                            pizzaorders.append(node["PIZZAORDER"])
                    if "DRINKORDER" in node:
                        if isinstance(node["DRINKORDER"], list):
                            drinkorders.extend(node["DRINKORDER"])
                        else:
                            drinkorders.append(node["DRINKORDER"])
                    if node.get("TYPE") == "PIZZAORDER":
                        pizzaorders.append(node)
                    if node.get("TYPE") == "DRINKORDER":
                        drinkorders.append(node)
            result = {}
            if pizzaorders:
                result["PIZZAORDER"] = pizzaorders
            if drinkorders:
                result["DRINKORDER"] = drinkorders
            if result:
                return {"ORDER": result}
            else:
                return {}

        elif key == "PIZZAORDER":
            number = None
            size = None
            style = None
            toppings = []
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "NUMBER":
                        number = node["VALUE"]
                    elif t == "SIZE":
                        size = node["VALUE"]
                    elif t == "STYLE":
                        style = node["VALUE"]
                    elif t == "TOPPING":
                        toppings.append(node)
            result = {}
            if number is not None:
                result["NUMBER"] = number
            if size is not None:
                result["SIZE"] = size
            if style is not None:
                result["STYLE"] = style
            if toppings:
                result["AllTopping"] = toppings
            # Mark type internally, will remove later
            result["TYPE"] = "PIZZAORDER"
            return result

        elif key == "DRINKORDER":
            number = None
            volume = None
            drinktype = None
            containertype = None
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "NUMBER":
                        number = node["VALUE"]
                    elif t == "VOLUME" or t == "SIZE":
                        volume = node["VALUE"]
                    elif t == "DRINKTYPE":
                        drinktype = node["VALUE"]
                    elif t == "CONTAINERTYPE":
                        containertype = node["VALUE"]
            result = {}
            if number is not None:
                result["NUMBER"] = number
            if volume is not None:
                result["SIZE"] = volume
            if drinktype is not None:
                result["DRINKTYPE"] = drinktype
            if containertype is not None:
                result["CONTAINERTYPE"] = containertype
            result["TYPE"] = "DRINKORDER"
            return result

        elif key in ["NUMBER","SIZE","STYLE","VOLUME","DRINKTYPE","CONTAINERTYPE","QUANTITY"]:
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            value_str = " ".join(values).strip()
            return {
                "TYPE": key,
                "VALUE": value_str
            }

        elif key == "TOPPING":
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            topping_str = " ".join(values).strip()
            return {
                "TYPE": "TOPPING",
                "NOT": False,
                "Quantity": None,
                "Topping": topping_str
            }

        elif key == "COMPLEX_TOPPING":
            quantity = None
            topping = None
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "QUANTITY":
                        quantity = node["VALUE"]
                    elif t == "TOPPING":
                        topping = node["Topping"]
            return {
                "TYPE": "TOPPING",
                "NOT": False,
                "Quantity": quantity,
                "Topping": topping
            }

        elif key == "NOT":
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict) and node.get("TYPE") == "TOPPING":
                    node["NOT"] = True
                    if "Quantity" not in node:
                        node["Quantity"] = None
                    return node
            return None

    else:
        # Try to parse sublists and combine orders found
        combined_order = {"PIZZAORDER": [], "DRINKORDER": []}
        found_order = False

        for el in cleaned:
            node = normalize_structure(el)
            if isinstance(node, dict):
                if "ORDER" in node:
                    found_order = True
                    order_node = node["ORDER"]
                    if "PIZZAORDER" in order_node:
                        combined_order["PIZZAORDER"].extend(order_node["PIZZAORDER"])
                    if "DRINKORDER" in order_node:
                        combined_order["DRINKORDER"].extend(order_node["DRINKORDER"])
                elif node.get("TYPE") == "PIZZAORDER":
                    found_order = True
                    combined_order["PIZZAORDER"].append(node)
                elif node.get("TYPE") == "DRINKORDER":
                    found_order = True
                    combined_order["DRINKORDER"].append(node)

        if found_order:
            final = {}
            if combined_order["PIZZAORDER"]:
                final["PIZZAORDER"] = combined_order["PIZZAORDER"]
            if combined_order["DRINKORDER"]:
                final["DRINKORDER"] = combined_order["DRINKORDER"]
            return {"ORDER": final} if final else {}

        return None
    
def normalize_structure2(tree):
    if not isinstance(tree, list):
        return None

    def is_key(token):
        return token in [
            "ORDER", "PIZZAORDER", "DRINKORDER", "NUMBER", "SIZE", "STYLE", "TOPPING",
            "QUANTITY", "VOLUME", "DRINKTYPE", "CONTAINERTYPE", "NOT", "COMPLEX_TOPPING"
        ]

    def remove_empty_orders(data):
        """
        Recursively remove empty PIZZAORDER or DRINKORDER nodes, but keep other fields.
        """
        if isinstance(data, dict):
            filtered = {}
            for k, v in data.items():
                if k in ["PIZZAORDER", "DRINKORDER"] and not v:  # Remove empty orders
                    continue
                filtered[k] = remove_empty_orders(v)
            return filtered
        elif isinstance(data, list):
            return [remove_empty_orders(item) for item in data]
        else:
            return data

    # Normalize tree
    cleaned = []
    for el in tree:
        cleaned.append(el)

    if len(cleaned) > 0 and isinstance(cleaned[0], str) and is_key(cleaned[0]):
        key = cleaned[0]
        if key == "ORDER":
            pizzaorders = []
            drinkorders = []
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    if "PIZZAORDER" in node:
                        pizzaorders.append(node["PIZZAORDER"])
                    elif "DRINKORDER" in node:
                        drinkorders.append(node["DRINKORDER"])
            result = {}
            if pizzaorders:
                result["PIZZAORDER"] = pizzaorders
            if drinkorders:
                result["DRINKORDER"] = drinkorders
            return remove_empty_orders({"ORDER": result}) if result else {}

        elif key == "PIZZAORDER":
            number = None
            size = None
            style = None
            toppings = []
            pending_quantity = None  # Track unassigned QUANTITY
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    t = node.get("TYPE")
                    if t == "NUMBER":
                        number = node["VALUE"]
                    elif t == "SIZE":
                        size = node["VALUE"]
                    elif t == "STYLE":
                        style = node["VALUE"]
                    elif t == "QUANTITY":
                        pending_quantity = node["VALUE"]  # Store the QUANTITY
                    elif t == "TOPPING":
                        # Attach pending QUANTITY to the current TOPPING
                        if pending_quantity:
                            node["Quantity"] = pending_quantity
                            pending_quantity = None
                        toppings.append(node)
                    elif t == "COMPLEX_TOPPING":
                        for topping in node["AllTopping"]:
                            toppings.append(topping)
            result = {}
            if number is not None:
                result["NUMBER"] = number
            if size is not None:
                result["SIZE"] = size
            if style is not None:
                result["STYLE"] = style
            if toppings:
                result["AllTopping"] = toppings
            return remove_empty_orders({"PIZZAORDER": result})

        elif key in ["NUMBER", "SIZE", "STYLE", "VOLUME", "DRINKTYPE", "CONTAINERTYPE", "QUANTITY"]:
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            value_str = " ".join(values).strip()
            return {"TYPE": key, "VALUE": value_str}

        elif key == "TOPPING":
            values = []
            for el in cleaned[1:]:
                if isinstance(el, str):
                    values.append(el)
            topping_str = " ".join(values).strip()
            return {"TYPE": "TOPPING", "NOT": False, "Quantity": None, "Topping": topping_str}

        elif key == "NOT":
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict) and node.get("TYPE") == "TOPPING":
                    node["NOT"] = True
                    if "Quantity" not in node:
                        node["Quantity"] = None
                    return node
            return None

        elif key == "COMPLEX_TOPPING":
            quantity = None
            topping = None
            for sub in cleaned[1:]:
                node = normalize_structure(sub)
                if isinstance(node, dict):
                    if node.get("TYPE") == "QUANTITY":
                        quantity = node["VALUE"]
                    elif node.get("TYPE") == "TOPPING":
                        topping = node["Topping"]
            if topping:
                return {"TYPE": "COMPLEX_TOPPING", "AllTopping": [{"Quantity": quantity, "Topping": topping}]}
            return None

    else:
        # Handle nested lists without specific keys
        combined_order = {"PIZZAORDER": [], "DRINKORDER": []}
        for el in cleaned:
            node = normalize_structure(el)
            if isinstance(node, dict):
                if "ORDER" in node:
                    order = node["ORDER"]
                    if "PIZZAORDER" in order:
                        combined_order["PIZZAORDER"].extend(order["PIZZAORDER"])
                    if "DRINKORDER" in order:
                        combined_order["DRINKORDER"].extend(order["DRINKORDER"])
                elif node.get("TYPE") == "PIZZAORDER":
                    combined_order["PIZZAORDER"].append(node)
                elif node.get("TYPE") == "DRINKORDER":
                    combined_order["DRINKORDER"].append(node)
        return remove_empty_orders({"ORDER": combined_order}) if combined_order["PIZZAORDER"] or combined_order["DRINKORDER"] else None

def remove_type_keys(obj):
    # Recursively remove "TYPE" keys from all dictionaries
    if isinstance(obj, dict):
        obj.pop("TYPE", None)
        for k, v in obj.items():
            remove_type_keys(v)
    elif isinstance(obj, list):
        for item in obj:
            remove_type_keys(item)


def preprocess_top(text):
    print(text)
    tokens = tokenize_top(text)
    parsed = parse_tokens(tokens)
    result = normalize_structure2(parsed)
    remove_type_keys(result)
    return result

def preprocess_true_top(text):
    print(text)
    tokens = tokenize_top(text)
    parsed = parse_tokens(tokens)
    result = normalize_structure(parsed)
    remove_type_keys(result)
    return result


def load_model(model, path):
    model.load_state_dict(torch.load(path))
    print(f"Model loaded from {path}")
    return model

def load_vocab():
    # loads the vocab from the text file "vocab.txt" and then swaps the key value pairs
    with open("../dataset2/vocab.txt", "r") as f:
        vocab = f.readlines()
    # remove any commas and single quotes
    vocab = [v.replace(",", "").replace("'", "") for v in vocab]
    vocab = {v.split(":")[0].strip():int(v.split(":")[1].strip()) for v in vocab}
    return vocab

vocab = load_vocab()
print(vocab)

class BiLSTMModel(nn.Module):
    def __init__(self, input_dim, embedding_dim, hidden_dim, output_dim, num_layers=3, dropout=0.5):
        super(BiLSTMModel, self).__init__()
        self.embedding = nn.Embedding(input_dim, embedding_dim, padding_idx=0)
        
        # Bidirectional LSTM
        self.bilstm_1 = nn.LSTM(
            input_size=embedding_dim,
            hidden_size=hidden_dim,
            num_layers=num_layers,
            bidirectional=True,
            batch_first=True,
            dropout=dropout,
        )

        # Fully connected layers
        self.fc2 = nn.Linear(hidden_dim * 2, output_dim)


    def forward(self, x):
        # Embedding layer
        embedded = self.embedding(x)

        # BiLSTM layer
        lstm_out, _ = self.bilstm_1(embedded)

        output = self.fc2(lstm_out)
        return F.log_softmax(output, dim=-1)
    

# Label maps for the models
MODEL_1_LABEL_MAP = {
    "B-PIZZAORDER": 1,
    "I-PIZZAORDER": 2,
    "B-DRINKORDER": 3,
    "I-DRINKORDER": 4,
    "O": 5
}

MODEL_2_LABEL_MAP = {
    'B-DRINKTYPE': 1, 'I-DRINKTYPE': 2,
    'B-SIZE': 3, 'I-SIZE': 4,
    'B-NUMBER': 5, 'I-NUMBER': 6,
    'B-CONTAINERTYPE': 7, 'I-CONTAINERTYPE': 8,
    'B-COMPLEX_TOPPING': 9, 'I-COMPLEX_TOPPING': 10,
    'B-TOPPING': 11, 'I-TOPPING': 12,
    'B-NEG_TOPPING': 13, 'I-NEG_TOPPING': 14,
    'B-NEG_STYLE': 15, 'I-NEG_STYLE': 16,
    'B-STYLE': 17, 'I-STYLE': 18,
    'B-QUANTITY': 19, 'I-QUANTITY': 20,
    'O': 21
}
def write_comparison_file(predicted_json, true_json, top_decoupled):
    with open("predicted_true.json", "a") as f:
        # write top decoupled 
        f.write(f"TOP_DECOUPLED: {top_decoupled}\n\n")
        # Format both JSONs side by side in a neat way
        predicted_str = json.dumps(predicted_json, indent=2)
        true_str = json.dumps(true_json, indent=2)

        max_width = max(len(line) for line in predicted_str.splitlines()) + 5
        max_lines = max(len(predicted_str.splitlines()), len(true_str.splitlines()))
        
        predicted_lines = predicted_str.splitlines()
        true_lines = true_str.splitlines()

        f.write(f"{'-' * (max_width * 2)}\n")
        f.write(f"{'Predicted'.center(max_width)}{'True'.center(max_width)}\n")
        f.write(f"{'-' * (max_width * 2)}\n")
        
        for i in range(max_lines):
            left = predicted_lines[i] if i < len(predicted_lines) else ""
            right = true_lines[i] if i < len(true_lines) else ""
            f.write(f"{left:<{max_width}}{right:<{max_width}}\n")
        
        f.write(f"{'-' * (max_width * 2)}\n")
        f.write(f"Match: {predicted_json == true_json}\n")
        f.write(f"{'-' * (max_width * 2)}\n\n")
# Function to apply both models and get the TOP_DECOUPLED format
def process_entry(entry):
    src_text = entry["train.SRC"]
    true_top = entry["train.TOP"]

    # Tokenize and preprocess the input text
    tokens = tokenize_input(src_text)
    # Convert tokens to integers
    tokens = tokens_to_ints(tokens, vocab)
    # Convert to tensor
    tokens = torch.tensor(tokens).unsqueeze(0).to(device)

    # Get predictions from the first model
    model_1_output = model_1(tokens)
    first_model_labels = model_1_output.argmax(dim=-1).squeeze(0).tolist()  # Ensure list of labels

    # Get predictions from the second model
    model_2_output = model_2(tokens)
    second_model_labels = model_2_output.argmax(dim=-1).squeeze(0).tolist()  # Ensure list of labels

    # Generate TOP_DECOUPLED output
    top_decoupled = generate_top_decoupled(src_text, first_model_labels, second_model_labels)

    # Preprocess TOP to JSON format
    predicted_json = preprocess_top(top_decoupled)
    true_json = preprocess_true_top(true_top)
    
    with open("predicted_true.json", "a") as f:
        f.write(f"tokens: {tokens}\n")
        f.write(f"true: {true_top}\n")
    write_comparison_file(predicted_json, true_json, top_decoupled)

    # Compare the predicted JSON with the ground truth JSON
    return predicted_json == true_json

def generate_top_decoupled(text, first_labels, second_labels):
    words = text.split()
    first_labels = first_labels[:len(words)]
    second_labels = second_labels[:len(words)]
    
    # Debugging output
    with open("predicted_true.json", "a") as f:
        f.write(str(words) + "\n")
        f.write(str([next(k for k, v in MODEL_1_LABEL_MAP.items() if v == l) for l in first_labels]) + "\n")
        f.write(str([next(k for k, v in MODEL_2_LABEL_MAP.items() if v == l) for l in second_labels]) + "\n\n") 
    
    
    result = ["(ORDER"]
    current_order_type = None
    current_group = None
    open_groups = []  # To keep track of open groups for proper closing

    for i, (word, first_label, second_label) in enumerate(zip(words, first_labels, second_labels)):
        first_label_key = next(
            (key for key, value in MODEL_1_LABEL_MAP.items() if value == first_label), None
        )
        # Handle the first labels (ORDER type: PIZZAORDER, DRINKORDER)
        if first_label in [MODEL_1_LABEL_MAP["B-PIZZAORDER"], MODEL_1_LABEL_MAP["B-DRINKORDER"]]:
            if current_order_type is not None:
                result.append(")")  # Close the previous order
                open_groups.pop()  # Remove from open_groups stack
            current_order_type = "PIZZAORDER" if first_label == MODEL_1_LABEL_MAP["B-PIZZAORDER"] else "DRINKORDER"
            result.append(f"({current_order_type}")
            open_groups.append(current_order_type)
            
        # if the first label is I- and the current order type is None, consider it as B- and add the order type
        # and do the same if it's an I- but for a different order type
        elif first_label_key.startswith("I-") and (current_order_type is None or current_order_type != first_label_key[2:]):
            if current_order_type is not None:
                result.append(")")  # Close the previous order
                open_groups.pop()
            current_order_type = "PIZZAORDER" if first_label == MODEL_1_LABEL_MAP["I-PIZZAORDER"] else "DRINKORDER"
            result.append(f"({current_order_type}")
            open_groups.append(current_order_type)
            

        elif first_label == MODEL_1_LABEL_MAP["O"] and current_order_type is not None:
            result.append(")")  # Close the current order
            open_groups.pop()
            current_order_type = None
            
        elif first_label == MODEL_1_LABEL_MAP["O"] and current_order_type is None:
            continue  # Skip the word if it's not part of an order

        # Handle the second labels (attributes like NUMBER, SIZE, TOPPING, etc.)
        if second_label != MODEL_2_LABEL_MAP["O"]:
            second_label_key = next(
                (key for key, value in MODEL_2_LABEL_MAP.items() if value == second_label), None
            )
            if not second_label_key:
                print(f"Warning: Unexpected label {second_label} encountered for word '{word}'. Skipping.")
                continue

            label_type = second_label_key.split("-")[-1]
            if label_type not in ["NEG_TOPPING", "NEG_STYLE"]:
                if second_label_key.startswith("B-"):
                    # Close the previous group if there is one
                    if current_group:
                        result.append(")")  # Close the previous group
                        # since this is positive, if the current top group is a not group close it as well
                        if current_group["type"] == "NEG_TOPPING" or current_group["type"] == "NEG_STYLE":
                            result.append(")")
                            open_groups.pop()
                        open_groups.pop()
                    current_group = {"type": label_type, "content": [word]}
                    result.append(f"({label_type} {word}")
                    open_groups.append(label_type)

                elif second_label_key.startswith("I-") and current_group and current_group["type"] == label_type:
                    current_group["content"].append(word)
                    result[-1] += f" {word}"  # Append to the last open group

                elif second_label_key.startswith("I-") and (not current_group or current_group["type"] != label_type):
                    print(f"Warning: I- tag '{label_type}' for word '{word}' without preceding B- tag. Converting to B-.")
                    # Close the previous group if there is one
                    if current_group:
                        result.append(")")  # Close the previous group
                        # since this is positive, if the current top group is a not group close it as well
                        if current_group["type"] == "NEG_TOPPING" or current_group["type"] == "NEG_STYLE":
                            result.append(")")
                            open_groups.pop()
                        open_groups.pop()
                    current_group = {"type": label_type, "content": [word]}
                    result.append(f"({label_type} {word}")
                    open_groups.append(label_type)

            # Special handling for NEG_TOPPING and NEG_STYLE
            else:
                if second_label_key.startswith("B-"):
                    if current_group:
                        result.append(")")
                        open_groups.pop()
                    result.append(f"(NOT ({'TOPPING' if label_type == 'NEG_TOPPING' else 'STYLE'} {word}")
                    current_group = {"type": label_type, "content": [word]}
                    open_groups.append(label_type)
                    open_groups.append("NOT")
                elif second_label_key.startswith("I-") and current_group and current_group["type"] == label_type:
                    current_group["content"].append(word)
                    result[-1] += f" {word}"  # Append to the last open group
                elif second_label_key.startswith("I-") and (not current_group or current_group["type"] != label_type):
                     # Close the previous group if there is one
                    if current_group:
                        result.append(")")
                        open_groups.pop()
                    print(f"Warning: I- tag '{label_type}' for word '{word}' without preceding B- tag. Converting to B-.")
                    result.append(f"(NOT ({'TOPPING' if label_type == 'NEG_TOPPING' else 'STYLE'} {word}")
                    current_group = {"type": label_type, "content": [word]}
                    open_groups.append("NOT")
                    open_groups.append(label_type)
        # Handle O labels
        else:
            if current_group:
                result.append(")")  # Close the current group
                open_groups.pop()
            current_group = None

    # Close any remaining open groups
    while open_groups:
        result.append(")")
        open_groups.pop()

    result.append(")")  # Close the overall ORDER group
    return " ".join(result)


def process_text_input(text, model_1, model_2, vocab, device):
    # Predict
    tokens = tokenize_input(text)
    tokens_int = tokens_to_ints(tokens, vocab)
    tokens_tensor = torch.tensor(tokens_int).unsqueeze(0).to(device)

    model_1_output = model_1(tokens_tensor)
    first_model_labels = model_1_output.argmax(dim=-1).squeeze(0).tolist()

    model_2_output = model_2(tokens_tensor)
    second_model_labels = model_2_output.argmax(dim=-1).squeeze(0).tolist()

    top_decoupled = generate_top_decoupled(text, first_model_labels, second_model_labels)
    predicted_json = preprocess_top(top_decoupled)
    return predicted_json, top_decoupled
# -----------------------------------------
# Streamlit Application
# -----------------------------------------
st.title("TOP Decoupled Prediction App")

st.write("Enter a sentence to generate a TOP-decoupled hierarchical representation.")

device = torch.device("cpu")
vocab = load_vocab()

input_dim = len(vocab)
embedding_dim = 128
hidden_dim = 128
output_dim1 = 6
output_dim2 = 22
num_layers = 2
dropout = 0.3

model_1 = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim1, num_layers, dropout).to(device)
model_2 = BiLSTMModel(input_dim, embedding_dim, hidden_dim, output_dim2, num_layers, dropout).to(device)

model_1 = load_model(model_1, "Bilstm_order_sequence.pt")
model_2 = load_model(model_2, "Bilstm_model2.pt")

input_text = st.text_area("Enter your text here:", "I would like one large thin crust pizza with hot cheese and pepperoni")

if st.button("Process Text"):
    predicted_json, top_decoupled = process_text_input(input_text, model_1, model_2, vocab, device)
    st.write("**TOP Decoupled:**")
    st.write(top_decoupled)
    st.write("**JSON Output:**")
    st.json(predicted_json)
