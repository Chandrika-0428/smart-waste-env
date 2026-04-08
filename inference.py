"""
inference.py — Smart Waste Management Environment
Baseline inference script for OpenEnv hackathon submission.

Calls the FastAPI endpoints (not the env directly).
Runs all 3 tasks: easy, medium, hard.
Prints exact required log format.

Environment variables:
  API_BASE_URL  — base URL of the running API (default: http://localhost:8000)
  MODEL_NAME    — agent model name label (default: random-baseline)
  HF_TOKEN      — Hugging Face token (optional, for authenticated spaces)
"""

import os
import random
import time
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "random-baseline")
HF_TOKEN = os.environ.get("HF_TOKEN", "")

HEADERS = {}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

TASK_MODES = ["easy", "medium", "hard"]
N_STEPS = 20
SEED = 42


def reset_env(task_mode: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_mode": task_mode, "seed": SEED},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(task_mode: str, action: int) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action, "task_mode": task_mode},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    assert "application/json" in resp.headers.get("content-type", ""), \
        f"Expected JSON response, got: {resp.headers.get('content-type')}"
    return resp.json()


def get_grade(task_mode: str) -> dict:
    resp = requests.get(
        f"{API_BASE_URL}/grade",
        params={"task_mode": task_mode},
        headers=HEADERS,
        timeout=10,
    )
    resp.raise_for_status()
    return resp.json()


def select_action(observation: dict, task_mode: str) -> int:
    """
    Heuristic baseline agent.
    Uses waste type to pick the most likely correct facility.
    """
    OPTIMAL = {0: 0, 1: 1, 2: 0, 3: 0, 4: 3, 5: 4}
    waste_type = observation.get("waste_type", 0)
    if random.random() < 0.2:
        return random.randint(0, 4)
    return OPTIMAL.get(waste_type, random.randint(0, 4))


def run_task(task_mode: str):
    rng = random.Random(SEED)

    print(f"[START] task={task_mode} model={MODEL_NAME} api={API_BASE_URL}")

    obs = reset_env(task_mode)

    total_reward = 0.0
    done = False
    step_num = 0

    while not done and step_num < N_STEPS:
        action = select_action(obs, task_mode)
        result = step_env(task_mode, action)

        obs = result["observation"]
        reward = result["reward"]
        done = result["done"]
        info = result["info"]

        step_num += 1
        total_reward += reward

        print(
            f"[STEP] step={step_num} "
            f"action={action} "
            f"facility={info['facility_chosen']} "
            f"waste={info['waste_type']} "
            f"correct={info['is_correct']} "
            f"reward={reward:.4f} "
            f"done={done}"
        )

    grade = get_grade(task_mode)

    print(
        f"[END] task={task_mode} "
        f"steps={step_num} "
        f"total_reward={total_reward:.4f} "
        f"avg_reward={total_reward/max(step_num,1):.4f} "
        f"grade={grade['score']:.4f} "
        f"grader={grade['grader']}"
    )
    print()


def wait_for_api(max_retries: int = 10, delay: float = 1.0):
    for attempt in range(max_retries):
        try:
            resp = requests.get(f"{API_BASE_URL}/health", timeout=5)
            if resp.status_code == 200:
                return True
        except requests.exceptions.ConnectionError:
            pass
        time.sleep(delay)
    raise RuntimeError(
        f"API at {API_BASE_URL} did not become ready after {max_retries} retries."
    )


if __name__ == "__main__":
    print(f"Connecting to API: {API_BASE_URL}")
    wait_for_api()
    print(f"API ready. Running inference with model={MODEL_NAME}\n")

    for task in TASK_MODES:
        run_task(task)

    print("All tasks complete.")