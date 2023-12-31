import csv

from datasets import load_dataset
from openai import OpenAI

client = OpenAI(api_key="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")


dataset = load_dataset("LabHC/moji", split="test")

row = ["id", "text", "label", "sa", "counter1", "conter2", "counter3", "counter4", "counter5"]

#with open("multi_counter_dataset.csv", "w", newline="") as file:
#    writer = csv.writer(file)
#    writer.writerow(row)

for i in range(len(dataset)):

    data = dataset[i]
    
    counter_texts = []
    
    for j in range(5):
        if data["sa"] == 0:
            prompt = data["text"] + " Translate this tweet to Standard-American English"
        else:
            prompt = data["text"] + " Translate this tweet to African-American English"
    
        response = client.chat.completions.create(
            model="gpt-4",
            messages=[{"role": "user", "content": prompt}]
        )
    
        counter_text = response.choices[0].message.content
        counter_texts.append(counter_text)
    
    row = [i, data["text"], data["label"], data["sa"], counter_texts[0], counter_texts[1], counter_texts[2],
           counter_texts[3], counter_texts[4]]
    with open("multi_counter_dataset.csv", "a", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(row)