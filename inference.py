import random
from src.waste_env import WasteManagementEnv

MAX_STEPS = 10


def log_start():
    print("[START] task=waste env=smart model=baseline", flush=True)


def log_step(step, action, reward, done):
    print(f"[STEP] step={step} action={action} reward={reward:.2f} done={str(done).lower()} error=null", flush=True)


def log_end(success, steps, rewards):
    rewards_str = ",".join(f"{r:.2f}" for r in rewards)
    print(f"[END] success={str(success).lower()} steps={steps} rewards={rewards_str}", flush=True)


def main():
    env = WasteManagementEnv(task_mode="hard")

    state, _ = env.reset()

    rewards = []
    log_start()

    for step in range(1, MAX_STEPS + 1):
        action = random.randint(0, 4)

        state, reward, done, _, _ = env.step(action)

        rewards.append(reward)

        log_step(step, action, reward, done)

        if done:
            break

    success = sum(rewards) > 1.0

    log_end(success, step, rewards)


if __name__ == "__main__":
    main()