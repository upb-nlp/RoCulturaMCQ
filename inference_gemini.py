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

def judge_item(prompt: str) -> dict:

    
    grounding_tool = types.Tool(
        google_search=types.GoogleSearch()
    )

    config = types.GenerateContentConfig(
        tools=[grounding_tool]
    )

    response = client.models.generate_content(
                model="gemini-3-pro-preview",
                contents=prompt,
                config=config
    )
    return response.text

def doc_to_text(doc):
    choices = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]   
    string = "Întrebare: {0}\nVariante:\n".format(doc["question"])
    for i, choice in enumerate(choices):
        string += "{1}. {0}\n".format(choice, str(chr(97 + i)).upper())
    string = string + "\n"
    string += "Răspunde direct cu litera opțiunii alese. Trebuie să răspunzi doar cu o literă: A, B, C sau D, chiar dacă răspunsul nu este clar.\nRăspuns: "
    return string


if __name__ == "__main__":

    


    dataset = json.load(open("rocult_closed.json", "r", encoding="utf-8"))
    gemini_answers = []
    aug = []
    data_index = 0
    for data in tqdm(dataset):
        if data_index < len(gemini_answers):
            aug.append(gemini_answers[data_index])
            data_index += 1
            continue
        data_index += 1
        prompt = doc_to_text(data)
        # print("Promptul pentru model: ", prompt)
        answer = judge_item(prompt)
        l = data.copy()
        l["gemini-3-pro-search_answer"] = answer
        # print("Răspunsul modelului: ", answer)
        # print("Răspunsul corect: ", data["answer"])
        # print("\n\n")
        aug.append(l)
        json.dump(aug, open("rocult_closed_gemini_search.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)
    
    json.dump(aug, open("rocult_closed_gemini_search.json", "w", encoding="utf-8"), indent=4, ensure_ascii=False)