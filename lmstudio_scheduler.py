from time import sleep, time
import requests
import json
import threading
from queue import Queue
from prompt_format_utils import format_prompt_messages_judge_item
import random
random.seed(42)

MAX_RETRIES = 5

workers = [
    {'ip': '10.89.11.40:1234', 'name': 'duti'},
    {'ip': '10.89.52.9:1234', 'name': 'stefan'},
    {'ip': '10.89.11.26:1234', 'name': 'mihai'},
    {'ip': '10.89.11.69:1234', 'name': 'alex'},
    {'ip': '10.89.51.239:1234', 'name': 'denis'},
    {'ip': '10.89.52.39:1234', 'name': 'razvan'},
    
]

job_queue = Queue()

MODEL_NAME = "qwen3.5-35b-a3b"

def send_request(session, url, request_dict, reasoning_enabled=False):
    messages = format_prompt_messages_judge_item(
        request_dict
    )

    system_prompt = next(
        (m["content"] for m in messages if m["role"] == "system"), None
    )
    user_input = next(
        (m["content"] for m in messages if m["role"] == "user"), ""
    )

    # Step 1: Native API — get reasoning with thinking enabled
    reasoning = ""
    if reasoning_enabled:
        reasoning_url = f"http://{url}/api/v1/chat"
        reasoning_payload = {
            "model": MODEL_NAME,
            "input": user_input,
            "system_prompt": system_prompt,
            #"reasoning": "on",
        }

        resp1 = session.post(reasoning_url, json=reasoning_payload)
        resp1.raise_for_status()
        result1 = resp1.json()

        for item in result1["output"]:
            if item["type"] == "reasoning":
                reasoning = item["content"]
                break

    # Step 2: Text completions — continue generation after </think>
    answer_url = f"http://{url}/v1/completions"
    # prompt = (
    #     f"<|im_start|>system\n{system_prompt}<|im_end|>\n"
    #     f"<|im_start|>user\n{user_input}<|im_end|>\n"
    #     f"<|im_start|>assistant\n<think>\n{reasoning}\n</think>\n\n"
    # )
    prompt = f"{system_prompt}\n\n{user_input}\n\n"
    answer_payload = {
        "model": MODEL_NAME,
        "prompt": prompt,
        "temperature": 0.7,
        #"max_tokens": 5,
        "stop": ["<|im_end|>"],
        "response_format": {
            "type": "json_schema",
            "json_schema": {
                "name": "answer",
                "strict": True,
                "schema": {
                    "type": "object",
                    "properties": {
                        "answer": {
                            "type": "string",
                            "enum": ["A", "B", "C", "D"],
                        },
                    },
                    "required": ["answer"],
                },
            },
        },
    }

    resp2 = session.post(answer_url, json=answer_payload)
    resp2.raise_for_status()
    result2 = resp2.json()

    answer = json.loads(result2["choices"][0]["text"])["answer"]

    return reasoning, answer

def worker_thread(worker):
    worker["successful_jobs"] = 0
    worker["failed_jobs"] = 0
    worker["total_time"] = 0.0
    worker["lock"] = threading.Lock()

    session = requests.Session()

    while True:
        job = job_queue.get()
        if job is None:
            job_queue.task_done()
            break

        try:
            start_time = time()
            reasoning, answer = send_request(session, worker["ip"], job, reasoning_enabled=False)
            elapsed = time() - start_time

            try:
                with open(
                    f"test_{worker['name']}_with_reasoning_{MODEL_NAME}.jsonl",
                    "a"
                ) as f:
                    job['reasoning'] = reasoning.strip()
                    job['response'] = answer.strip()
                    json.dump(job, f)
                    f.write("\n")

                with worker["lock"]:
                    worker["successful_jobs"] += 1
                    worker["total_time"] += elapsed

            except json.JSONDecodeError:
                job["retries"] += 1
                if job["retries"] <= MAX_RETRIES:
                    job_queue.put(job)
                    sleep(1)
                else:
                    with worker["lock"]:
                        worker["failed_jobs"] += 1

        except Exception as e:
            print(f"[Job {job['id']}] Error on {worker['name']}: {e}")
            job_queue.put(job)
            with worker["lock"]:
                worker["failed_jobs"] += 1
            sleep(60)

        finally:
            job_queue.task_done()

if __name__ == "__main__":
    dataset = json.load(open("rocult_unified_answers.json"))
    for i, data in enumerate(dataset):
        data["id"] = i
    random.shuffle(dataset)


    start_time_global = time()

    threads = []
    for worker in workers:
        t = threading.Thread(target=worker_thread, args=(worker,))
        t.start()
        threads.append(t)

    for item in dataset:
        item["retries"] = 0
        job_queue.put(item)

    total_jobs = len(dataset)

    while job_queue.unfinished_tasks > 0:
        remaining = job_queue.unfinished_tasks

        total_done = 0
        total_time_spent = 0.0

        print("\nJobs remaining:", remaining)

        for worker in workers:
            with worker["lock"]:
                total_done += worker["successful_jobs"]
                total_time_spent += worker["total_time"]

                avg_time = (
                    worker["total_time"] / worker["successful_jobs"]
                    if worker["successful_jobs"] > 0 else 0
                )

                print(
                    f"  Worker {worker['name']}: "
                    f"{worker['successful_jobs']} successful, "
                    f"{worker['failed_jobs']} failed, "
                    f"avg {avg_time:.2f}s/request"
                )

        # -------- ETA calculation --------
        if total_done > 0:
            global_avg = total_time_spent / total_done
            eta_seconds = (remaining * global_avg) / len(workers)
            eta_minutes = eta_seconds / 60
            print(f"Estimated remaining time: {eta_minutes:.1f} minutes")
        else:
            print("Estimated remaining time: calculating...")

        sleep(5)

    job_queue.join()

    print("\nFinal stats:")
    for worker in workers:
        with worker["lock"]:
            avg_time = (
                worker["total_time"] / worker["successful_jobs"]
                if worker["successful_jobs"] > 0 else 0
            )
            print(
                f"  Worker {worker['name']}: "
                f"{worker['successful_jobs']} successful, "
                f"{worker['failed_jobs']} failed, "
                f"avg {avg_time:.2f}s/request"
            )

    for _ in workers:
        job_queue.put(None)

    for t in threads:
        t.join()
