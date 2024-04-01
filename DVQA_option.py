import json
import random
from tqdm import tqdm

english_number_to_numeric = {
    "zero": 0, "one": 1, "two": 2, "three": 3, "four": 4,
    "five": 5, "six": 6, "seven": 7, "eight": 8, "nine": 9,
    "ten": 10, "eleven": 11, "twelve": 12, "thirteen": 13,
    "fourteen": 14, "fifteen": 15, "sixteen": 16, "seventeen": 17,
    "eighteen": 18, "nineteen": 19,
    "twenty": 20, "thirty": 30, "forty": 40, "fifty": 50,
    "sixty": 60, "seventy": 70, "eighty": 80, "ninety": 90
}

for tens in ["twenty", "thirty", "forty", "fifty", "sixty", "seventy", "eighty", "ninety"]:
    for ones in range(10):
        english_number_to_numeric[f"{tens}{'' if ones == 0 else ' ' + str(ones)}"] = english_number_to_numeric[tens.split()[0]] + ones

def english_to_arabic(english_number):
    return english_number_to_numeric.get(english_number, english_number)

def load_json(filename):
    with open(filename, 'r') as file:
        data = json.load(file)
    if 'metadata' in filename:
        data = {item['image']: item['table'] for item in data}
    return data

def save_json(data, filename):
    with open(filename, 'w') as file:
        json.dump(data, file, indent=4)

def to_number(s):
    try:
        return int(s)
    except ValueError:
        try:
            return float(s)
        except ValueError:
            return None  

def find_table_by_image_name(image_name, meta_data):
    return meta_data.get(image_name, None)

def get_numeric_options(correct_answer, table):
    numeric_options = set(sum([[str(item) for item in sublist] for sublist in table[1:]], []))
    numeric_options = {option for option in numeric_options if isinstance(to_number(option), (int, float)) and option != correct_answer}
    while len(numeric_options) < 3:
        additional_option = str(random.randint(1, 10))
        if additional_option not in numeric_options and additional_option != correct_answer:
            numeric_options.add(additional_option)
    return random.sample(numeric_options, min(len(numeric_options), 3))

def get_english_number_options(correct_answer):
    correct_answer_numeric = english_to_arabic(correct_answer)
    options = [i for i in range(max(0, correct_answer_numeric-2), max(0, correct_answer_numeric-2)+4) if i != correct_answer_numeric]
    return list(map(str, options))

def get_other_options(correct_answer, table):
    options = []
    for option in table[0]:
        if option is not None and option.lower() != correct_answer.lower():
            options.append(option)
    return random.sample(options, min(len(options), 3))

def main():
    train_data = load_json('/data/luogen_code/LLaVA-robust/playground/data/DVQA/train.json')
    meta_data = load_json('/data/luogen_code/LLaVA-robust/playground/data/DVQA/metadata/train_metadata.json')

    for item in tqdm(train_data):
        image_name = item['image'].split('/')[-1]
        table = find_table_by_image_name(image_name, meta_data)
        if table is None:
            continue

        correct_answer = item['conversations'][1]['value']
        if correct_answer.lower() in english_number_to_numeric:
            correct_answer = str(english_to_arabic(correct_answer.lower()))

        options = []
        if to_number(correct_answer) is not None:
            options = get_numeric_options(correct_answer, table)
        elif correct_answer.lower() in ["yes", "no"]:
            options = ["yes" if correct_answer.lower() == "no" else "no"]
        elif correct_answer.lower() in english_number_to_numeric:
            options = get_english_number_options(correct_answer)
        else:
            options = get_other_options(correct_answer, table)

        if correct_answer not in options:
            options.append(correct_answer)
        random.shuffle(options)

        letters = ['A', 'B', 'C', 'D'][:len(options)]
        correct_option_letter = letters[options.index(correct_answer)]

        question_options_str = "\n".join([f"{letter}. {option}" for letter, option in zip(letters, options)])
        item['conversations'][0]['value'] += "\n" + question_options_str + "\nAnswer with the option's letter from the given choices directly."
        item['conversations'][1]['value'] = correct_option_letter

    save_json(train_data, 'updated_train.json')

if __name__ == "__main__":
    main()
