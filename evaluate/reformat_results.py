import json
import csv


def keys_to_upper(sub_tree: dict):
    sub_tree = {key.upper(): value for key, value in sub_tree.items()}
    return sub_tree


def process_pizzas(pizza_list):
    output_str = ""
    for pizza in pizza_list:
        output_str += "(PIZZAORDER "
        pizza = keys_to_upper(pizza)

        if "NUMBER" in pizza.keys() and pizza["NUMBER"]:
            output_str += f"(NUMBER {pizza['NUMBER']} ) "

        if "SIZE" in pizza.keys() and pizza['SIZE']:
            output_str += f"(SIZE {pizza['SIZE']} ) "

        if "STYLE" in pizza.keys() and pizza['STYLE']:
            for style in pizza['STYLE']:
                style = keys_to_upper(style)
                negated_style = False
                if "NOT" in style.keys() and style['NOT']:
                    output_str += "(NOT "
                    negated_style = True
                if "TYPE" in style.keys() and style['TYPE']:
                    output_str += f"(STYLE {style['TYPE']} ) "
                if negated_style:
                    output_str += ") "

        if "ALLTOPPING" in pizza.keys() and pizza['ALLTOPPING']:
            for topping in pizza['ALLTOPPING']:
                topping = keys_to_upper(topping)

                negated_topping = False
                if "NOT" in topping.keys() and topping['NOT']:
                    output_str += "(NOT "
                    negated_topping = True

                if "QUANTITY" in topping.keys() and topping['QUANTITY']:
                    output_str += f"(COMPLEX_TOPPING (QUANTITY {topping['QUANTITY']} ) "
                    if "TOPPING" in topping.keys() and topping['TOPPING']:
                        output_str += f"(TOPPING {topping['TOPPING']} ) "
                    output_str += ") "

                elif "TOPPING" in topping.keys() and topping['TOPPING']:
                    output_str += f"(TOPPING {topping['TOPPING']} ) "

                if negated_topping:
                    output_str += ") "
        output_str += ") "
    return output_str


def process_drinks(drinks_list):
    output_str = ""
    for drink in drinks_list:
        output_str += "(DRINKORDER "
        drink = keys_to_upper(drink)

        if "NUMBER" in drink.keys() and drink["NUMBER"]:
            output_str += f"(NUMBER {drink['NUMBER']} ) "

        if "SIZE" in drink.keys() and drink['SIZE']:
            output_str += f"(SIZE {drink['SIZE']} ) "

        if "DRINKTYPE" in drink.keys() and drink['DRINKTYPE']:
            output_str += f"(DRINKTYPE {drink['DRINKTYPE']} ) "

        if "VOLUME" in drink.keys() and drink['VOLUME']:
            output_str += f"(VOLUME {drink['VOLUME']} ) "

        if "CONTAINERTYPE" in drink.keys() and drink['CONTAINERTYPE']:
            output_str += f"(CONTAINERTYPE {drink['CONTAINERTYPE']} ) "

        output_str += ") "
    return output_str


def parse_tree(tree: dict):
    if tree is None:
        return ""  # Or handle this case appropriately
    tree = keys_to_upper(tree)
    output_str = ""
    if "ORDER" in tree.keys():
        output_str += "(ORDER "
        tree = tree['ORDER']

        tree = keys_to_upper(tree)

        pizzas = []
        if 'PIZZAORDER' in tree.keys():
            pizzas = tree['PIZZAORDER']

        drinks = []
        if 'DRINKORDER' in tree.keys():
            drinks = tree['DRINKORDER']

        output_str += process_pizzas(pizzas)

        output_str += process_drinks(drinks)

        output_str += ")"
    return output_str



# Driver code:
def main():
    input_file_path = "evaluate/json_output.json"
    output_file_path = "1.csv"

    with open(input_file_path, 'r') as j_file:
        json_data = json.load(j_file)

    # Prepare the output CSV file
    with open(output_file_path, mode='w', newline='') as csv_file:
        csv_writer = csv.writer(csv_file)
        # Write the header row
        csv_writer.writerow(["id", "output"])

        # Process each JSON object and write the results
        for idx, json_object in enumerate(json_data):
            print (f"Processing JSON object {idx}")
            print (json_object)
            parsed_output = parse_tree(json_object)
            csv_writer.writerow([idx, parsed_output])

    print(f"Parsing complete. Results saved to {output_file_path}.")


if __name__ == "__main__":
    main()
