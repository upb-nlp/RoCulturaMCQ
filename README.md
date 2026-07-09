# RoCulturaMCQ

A Romanian‑culture multiple‑choice benchmark: **1,355 four‑option questions** across **18 cultural categories** (history, gastronomy, language & idioms, legends & superstitions, politics, showbiz, music, sport, jokes, riddles, …), each with a gold answer and predictions from **108 models**.

Models are prompted to answer as a native Romanian speaker, replying with a single letter (A–D).

## Data

`rocult_unified_answers.json` — one object per question:

```json
{
  "category": "Istorie",
  "question": "...",
  "option_a": "...", "option_b": "...", "option_c": "...", "option_d": "...",
  "answer": "A",                       // gold answer
  "claude_opus_4_6_answer": "A",       // one field per evaluated model
  "gpt-oss-20b_high_reasoning": "...", // reasoning models also store their trace
  ...
}
```

`estimates.xlsx` / `estimates_22.csv` — per‑model accuracy (overall + per‑category, micro & macro).

## Scripts

| File | Purpose |
|------|---------|
| `inference_{claude,gemini,gpt}.py` | Run a closed‑source model over the benchmark; writes its answer back into the JSON. |
| `inference_{claude,gemini,gpt}_classification.py` | Use a model as a judge to flag question quality (valid / no definitive answer / bad distractor / wrong answer / bad question). |
| `lmstudio_scheduler.py` | Distribute open‑weight inference across a pool of LM Studio servers via a job queue (optional reasoning + JSON‑schema answers). |

## Evaluating open models (lm-evaluation-harness)

The benchmark is integrated into the [OpenLLM-Ro fork of lm-evaluation-harness](https://github.com/OpenLLM-Ro/lm-evaluation-harness) as the **`ro_cult`** task. It loads the dataset from the Hugging Face the same `question` / `option_a`–`option_d` / `answer` fields as this repo — and scores it as a 0‑shot `multiple_choice` task with the same prompt used by the inference scripts above.

```bash
git clone https://github.com/OpenLLM-Ro/lm-evaluation-harness
cd lm-evaluation-harness && pip install -e .

lm_eval --model hf \
  --model_args pretrained=OpenLLM-Ro/RoLlama3.1-8b-Instruct \
  --tasks ro_cult --num_fewshot 0 --batch_size auto --device cuda:0
```

Reported metric is accuracy (`acc`), matching the *Micro Acc.* column in `estimates.xlsx` / `estimates_22.csv`.

