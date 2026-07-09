import json
import sys
from tqdm import tqdm
from google import genai
from google.genai.types import GenerateContentConfig, ThinkingConfig
from google.genai import types
from dotenv import load_dotenv, find_dotenv
from pathlib import Path
import os
load_dotenv(".env")

client = genai.Client(api_key=os.getenv("GEMINI_API_KEY"))

MODEL_NAME = "gemini-3-pro-preview"
LABEL_FIELD = f"class_{MODEL_NAME}"

SYSTEM_PROMPT = """
Ești un evaluator de întrebări tip grilă.

Trebuie să analizezi întrebarea, variantele de răspuns și varianta de răspuns marcat drept corect din câmpul „answer” și să alegi EXACT una dintre următoarele etichete:

1. "No definitive answer" – Nu se poate determina dacă răspunsul corect este într-adevăr corect sau întrebarea nu are un răspuns clar.
2. "Valid question and answers" – Întrebarea și toate variantele sunt corecte, iar răspunsul marcat este corect.
3. "Invalid distractor" – Cel puțin una dintre variantele greșite ar putea fi corectă.
4. "Wrong correct answer" – Răspunsul marcat ca fiind corect este de fapt greșit.
5. "Wrong question" – Întrebarea este greșită, extrem de vagă, face referire la ceva de nu există, sau nu are sens. 

Răspunde DOAR cu eticheta (1, 2, 3, 4 sau 5).
"""


def classify(item):

    
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
                model=MODEL_NAME,
                contents=SYSTEM_PROMPT + build_prompt(item),
                config=config
    )
    return response.text

def build_prompt(item):
    return f"""
Analizează următoarea întrebare:

{format_question(item)}
"""


def format_question(item):
    return f"""{item['question']}

A. {item['option_a']}
B. {item['option_b']}
C. {item['option_c']}
D. {item['option_d']}

Răspuns corect: {item['answer']}"""


def process_json(input_json):
    results = []

    for item_id, item in enumerate(input_json):
        if LABEL_FIELD in item and item[LABEL_FIELD] != None:
            print(f"Item {item_id} already has label {item[LABEL_FIELD]}, skipping.")
            results.append(item)
            continue
        if item_id % 10 == 0 or item_id == 1:
            print(f"Processing item {item_id}/{len(input_json)}")
            json.dump(input_json, open("rocult_classification_gemini.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
        label = classify(item)
        # adaugă eticheta în JSON
        item[LABEL_FIELD] = label
        results.append(item)

    return results

if __name__ == "__main__":

    dataset = json.load(open("rocult_classification.json", "r", encoding="utf-8"))
    output = process_json(dataset)
    json.dump(dataset, open("rocult_classification_gemini.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)