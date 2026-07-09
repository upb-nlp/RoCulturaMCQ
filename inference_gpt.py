import os
from dotenv import load_dotenv
load_dotenv()
from openai import OpenAI
import json
from datasets import load_dataset
from tqdm import tqdm
client = OpenAI(
    api_key=os.getenv("OPENAI_API_KEY")
)

SYSTEM_PROMPT = """"Ești un asistent care răspunde la întrebări din perspectiva unui vorbitor nativ din România. Interpretează fiecare cerință folosind, atunci când este relevant, contextul cultural românesc: limbaj și uz local, instituții și administrație din România, educație/curriculum, istorie și geografie, mass-media și cultură populară, obiceiuri și sărbători, monedă (RON/leu), unități de măsură, fus orar (EET/EEST) și formatul datelor (zi/lună/an).

Respectarea instrucțiunilor are prioritate absolută:
- Urmează cu strictețe formatul de răspuns și regulile din promptul utilizatorului (de ex. să alegi dintre opțiuni, să răspunzi cu o singură opțiune, să nu adaugi explicații).  
- Nu inventa informații sau opțiuni și nu depăși cerințele de format. 

Când apar ambiguități:
- Alege interpretarea cea mai probabilă pentru un român (România ca referință implicită), fără a forța ancorarea dacă întrebarea indică clar alt context.
- referă opțiunea care este cea mai plauzibilă și corectă în context românesc, menținând consecvența cu uzul local."""

def judge_item(prompt: str) -> dict:
    response = client.beta.chat.completions.parse(
        model="gpt-5.2",
        messages=[
            {"role": "system", "content": SYSTEM_PROMPT},
            {"role": "user", "content": f"{prompt}"}
        ],
        seed=42
    )
    text = response.choices[0].message.content

    return text.strip()

def doc_to_text(doc):
    choices = [doc["option_a"], doc["option_b"], doc["option_c"], doc["option_d"]]
    
    string = "Întrebare: {0}\nVariante:\n".format(doc["question"])
    for i, choice in enumerate(choices):
        string += "{1}. {0}\n".format(choice, str(chr(97 + i)).upper())
    # string = string[:-1]
    string = string + "\n"
    string += "Răspunde direct cu litera opțiunii alese. Trebuie să răspunzi doar cu o literă: A, B, C sau D, chiar dacă răspunsul nu este clar.\nRăspuns: "
    return string



if __name__ == "__main__":
    dataset = json.load(open("rocult_unified_answers.json", "r"))
    for data in tqdm(dataset):
        prompt = doc_to_text(data)
        answer = judge_item(prompt)
        data["gpt5.2_sp_answer"] = answer

    json.dump(dataset, open("rocult_unified_answers.json", "w"), indent=4, ensure_ascii=False)