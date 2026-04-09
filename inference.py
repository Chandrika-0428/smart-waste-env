import os
import random
import time
import requests

API_BASE_URL = os.environ.get("API_BASE_URL", "http://localhost:8000").rstrip("/")
MODEL_NAME = os.environ.get("MODEL_NAME", "random-baseline")
HF_TOKEN = os.environ.get("HF_TOKEN")

HEADERS = {"Content-Type": "application/json"}
if HF_TOKEN:
    HEADERS["Authorization"] = f"Bearer {HF_TOKEN}"

TASK_MODES = ["easy", "medium", "hard"]
N_STEPS = 20
SEED = 42

OPTIMAL = {0: 0, 1: 1, 2: 0, 3: 0, 4: 3, 5: 4}


def reset_env(task_mode: str) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/reset",
        json={"task_mode": task_mode, "seed": SEED},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def step_env(task_mode: str, action: int) -> dict:
    resp = requests.post(
        f"{API_BASE_URL}/step",
        json={"action": action, "task_mode": task_mode},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def get_grade(task_mode: str) -> dict:
    resp = requests.get(
        f"{API_BASE_URL}/grade",
        params={"task_mode": task_mode},
        headers=HEADERS,
        timeout=15,
    )
    resp.raise_for_status()
    return resp.json()


def select_action(observation: dict) -> int:
    obs_data = observation.get("observation", observation)

    if isinstance(obs_data, dict):
        waste_type = obs_data.get("waste_type", 0)
    else:
        waste_type = round(obs_data[0] * 5) if obs_data else 0

    if random.random() < 0.15:
        return random.randint(0, 4)

    return OPTIMAL.get(int(waste_type), random.randint(0, 4))


def run_task(task_mode: str):
    rewards = []
    error_msg = None

    print(f"[START] task={task_mode} env=smart-waste-management-env model={MODEL_NAME}")

    try:
        result = reset_env(task_mode)
        obs = result.get("observation", result)

        done = False
        step_num = 0

        while not done and step_num < N_STEPS:
            action = select_action(obs)

            try:
                step_result = step_env(task_mode, action)
                obs = step_result.get("observation", {})
                reward = float(step_result.get("reward", 0.0))
                done = bool(step_result.get("done", False))

                rewards.append(reward)
                step_num += 1

                print(
                    f"[STEP] step={step_num} "
                    f"action={action} "
                    f"reward={reward:.2f} "
                    f"done={str(done).lower()} "
                    f"error=null"
                )

            except Exception as e:
                error_msg = str(e)
                rewards.append(0.0)
                step_num += 1

                print(
                    f"[STEP] step={step_num} "
                    f"action={action} "
                    f"reward=0.00 "
                    f"done=false "
                    f"error={error_msg}"
                )
                break

    except Exception as e:
        error_msg = str(e)

    grade = get_grade(task_mode) if error_msg is None else {"score": 0.0}
    score = float(grade.get("score", 0.0))
    score = max(0.0, min(score, 1.0))

    success = score >= 0.1
    steps = len(rewards)
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)

    print(
        f"[END] success={str(success).lower()} "
        f"steps={steps} "
        f"score={score:.3f} "
        f"rewards={rewards_str}"
    )


if __name__ == "__main__":
    for task in TASK_MODES:
        run_task(task)