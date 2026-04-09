"""
Smart Waste Management RL Environment
"""
import numpy as np
import random

# Waste types
WASTE_TYPES = {
    0: "plastic",
    1: "organic",
    2: "glass",
    3: "metal",
    4: "hazardous",
    5: "general",
}

# Facility names
FACILITIES = {
    0: "recycling_plant",
    1: "compost_facility",
    2: "landfill",
    3: "hazardous_facility",
    4: "energy_recovery",
}

# Correct facilities per waste type (primary, secondary)
CORRECT_FACILITIES = {
    0: [0, 4],   # plastic → recycling, energy recovery
    1: [1],      # organic → compost
    2: [0],      # glass   → recycling
    3: [0],      # metal   → recycling
    4: [3],      # hazardous → hazardous facility
    5: [4, 2],   # general → energy recovery, landfill
}

# Cost matrix [waste_type][facility] — lower is better (0..1)
COST_MATRIX = np.array([
    [0.1, 0.9, 0.7, 0.8, 0.3],  # plastic
    [0.9, 0.1, 0.6, 0.9, 0.5],  # organic
    [0.1, 0.9, 0.7, 0.8, 0.5],  # glass
    [0.1, 0.9, 0.6, 0.7, 0.4],  # metal
    [0.9, 0.9, 0.8, 0.1, 0.9],  # hazardous
    [0.5, 0.8, 0.4, 0.9, 0.2],  # general
], dtype=np.float32)

# CO2 impact matrix [waste_type][facility] — lower is better
CO2_MATRIX = np.array([
    [0.1, 0.8, 0.9, 0.7, 0.3],  # plastic
    [0.8, 0.1, 0.7, 0.9, 0.4],  # organic
    [0.1, 0.8, 0.9, 0.7, 0.4],  # glass
    [0.1, 0.8, 0.8, 0.7, 0.3],  # metal
    [0.9, 0.9, 0.9, 0.2, 0.9],  # hazardous
    [0.5, 0.7, 0.6, 0.9, 0.2],  # general
], dtype=np.float32)


class WasteManagementEnv:
    def __init__(self, task_mode: str = "easy", seed: int = 42):
        self.task_mode = task_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)

        self.max_steps = 20
        self.step_count = 0
        self.done = False
        self.co2_budget = 1.0

        # Facility capacities
        self.capacities = np.ones(5, dtype=np.float32)

        self._current_waste = None
        self._current_obs = None

        self.history = []  # list of (action, reward, correct)
        self._generate_waste()

    def _generate_waste(self):
        waste_type = self.rng.randint(0, 5)
        quantity = round(self.rng.uniform(0.1, 1.0), 2)
        zone = self.rng.randint(0, 2)
        time_of_day = round(self.rng.uniform(0.0, 1.0), 2)
        recycling_rate = round(self.rng.uniform(0.3, 0.9), 2)

        self._current_waste = {
            "waste_type": waste_type,
            "quantity": quantity,
            "zone": zone,
            "time_of_day": time_of_day,
            "recycling_rate": recycling_rate,
        }
        self._current_obs = self._build_obs()

    def _build_obs(self) -> list:
        w = self._current_waste
        obs = [
            w["waste_type"] / 5.0,
            w["quantity"],
            w["zone"] / 2.0,
            w["time_of_day"],
            w["recycling_rate"],
            float(self.capacities[0]),
            float(self.capacities[1]),
            float(self.capacities[2]),
            float(self.capacities[3]),
            float(self.capacities[4]),
            self.step_count / self.max_steps,
            self.co2_budget,
        ]
        return [round(float(x), 4) for x in obs]

    def _observation_dict(self) -> dict:
        w = self._current_waste
        return {
            "observation": self._current_obs,
            "waste_type": w["waste_type"],
            "waste_name": WASTE_TYPES[w["waste_type"]],
            "quantity": w["quantity"],
            "zone": w["zone"],
            "time_of_day": w["time_of_day"],
            "recycling_rate": w["recycling_rate"],
            "step": self.step_count,
            "co2_budget": round(self.co2_budget, 4),
        }

    def reset(self) -> dict:
        self.__init__(self.task_mode, self.seed)
        return self._observation_dict()

    def step(self, action: int) -> dict:
        if self.done:
            raise ValueError("Episode done. Call reset().")

        waste_type = self._current_waste["waste_type"]
        correct_facilities = CORRECT_FACILITIES[waste_type]
        is_correct = action in correct_facilities

        # --- Correctness score ---
        correctness = 1.0 if is_correct else 0.0
        # partial credit: landfill always gets 0.1
        if not is_correct and action == 2:
            correctness = 0.1

        # --- Cost efficiency score (1 - normalized cost) ---
        cost_score = 1.0 - float(COST_MATRIX[waste_type][action])

        # --- CO2 score (1 - normalized co2 impact) ---
        co2_impact = float(CO2_MATRIX[waste_type][action])
        co2_score = 1.0 - co2_impact

        # --- Capacity score ---
        cap_score = float(self.capacities[action])

        # --- Composite reward by task ---
        if self.task_mode == "easy":
            reward = correctness
        elif self.task_mode == "medium":
            reward = 0.6 * correctness + 0.4 * cost_score
        else:  # hard
            reward = (
                0.4 * correctness
                + 0.3 * cost_score
                + 0.2 * co2_score
                + 0.1 * cap_score
            )

        reward = float(np.clip(reward, 0.0, 1.0))

        # Update state
        self.capacities[action] = max(0.0, self.capacities[action] - 0.05)
        self.co2_budget = max(0.0, self.co2_budget - co2_impact * 0.05)
        self.step_count += 1
        self.done = self.step_count >= self.max_steps

        self.history.append({
            "action": action,
            "waste_type": waste_type,
            "is_correct": is_correct,
            "reward": reward,
        })

        info = {
            "facility_chosen": FACILITIES[action],
            "waste_type": WASTE_TYPES[waste_type],
            "is_correct": is_correct,
            "correctness": correctness,
            "cost_score": round(cost_score, 4),
            "co2_score": round(co2_score, 4),
            "cap_score": round(cap_score, 4),
        }

        if not self.done:
            self._generate_waste()

        return {
            "observation": self._observation_dict(),
            "reward": round(reward, 4),
            "done": self.done,
            "info": info,
        }

    def grade(self) -> dict:
        if not self.history:
            return {"score": 0.0, "grader": self._grader_name()}

        rewards = [h["reward"] for h in self.history]
        score = float(np.clip(np.mean(rewards), 0.0, 1.0))

        return {
            "score": round(score, 4),
            "grader": self._grader_name(),
            "steps": len(self.history),
            "correct_count": sum(1 for h in self.history if h["is_correct"]),
        }

    def _grader_name(self) -> str:
        return {
            "easy": "correctness_grader",
            "medium": "efficiency_grader",
            "hard": "sustainability_grader",
        }[self.task_mode]

    def state(self) -> dict:
        return {
            "task_mode": self.task_mode,
            "step": self.step_count,
            "done": self.done,
            "co2_budget": round(self.co2_budget, 4),
            "capacities": {FACILITIES[i]: round(float(self.capacities[i]), 4) for i in range(5)},
            "current_waste": self._current_waste,
            "observation": self._current_obs,
        }