import json
import spacy
from tqdm import tqdm

nlp = spacy.load("en_core_web_sm")

with open('/data/vldata/anns/blip_laion_cc_sbu_558k.json', 'r', encoding='utf-8') as source_file:
    source_data = json.load(source_file)

count = 0
target_data = []

for item in tqdm(source_data):
    conversations = item["conversations"]
    
    for conversation in conversations:
        if conversation["from"] == "gpt":
            doc = nlp(conversation["value"])
            noun_phrases = [chunk.text for chunk in doc.noun_chunks]

            split_item = {
                "caption": '.'.join(noun_phrases) + '.',  
                "iid": count,
                "id": item["id"],
                "url": "/hdd1/image/images/" + item["image"],  
                "sentence": conversation["value"],  
                "from": conversation["from"]
            }
            count += 1
            target_data.append(split_item)

with open('output1.json', 'w', encoding='utf-8') as file:
    json.dump(target_data, file, indent=4)

print("JSON file updated successfully.")
