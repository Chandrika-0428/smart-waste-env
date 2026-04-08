import numpy as np
import random
from typing import Dict, Any, Tuple, List

WASTE_TYPES = {
    0: "plastic",
    1: "organic",
    2: "glass",
    3: "metal",
    4: "hazardous",
    5: "general",
}

FACILITIES = {
    0: "recycling_plant",
    1: "compost_facility",
    2: "landfill",
    3: "hazardous_facility",
    4: "energy_recovery",
}

CORRECT_FACILITY = {
    0: [0, 4],
    1: [1, 2],
    2: [0, 2],
    3: [0, 2],
    4: [3],
    5: [2, 4],
}

OPTIMAL_FACILITY = {
    0: 0,
    1: 1,
    2: 0,
    3: 0,
    4: 3,
    5: 4,
}

BASE_COST = {
    0: 0.10,
    1: 0.08,
    2: 0.15,
    3: 0.60,
    4: 0.12,
}

ZONE_DISTANCE_MULTIPLIER = {0: 1.0, 1: 1.4, 2: 1.9}

CO2_EMISSION = {
    (0, 0): 0.1,
    (0, 2): 0.8,
    (0, 4): 0.3,
    (1, 1): 0.05,
    (1, 2): 0.9,
    (2, 0): 0.1,
    (2, 2): 0.5,
    (3, 0): 0.1,
    (3, 2): 0.6,
    (4, 3): 0.0,
    (4, 2): 1.5,
    (5, 2): 0.4,
    (5, 4): 0.2,
}

MAX_STEPS = 20
MAX_CAPACITY = 500.0


class WasteManagementEnv:
    def __init__(self, task_mode: str = "easy", seed: int = 42):
        self.task_mode = task_mode
        self.seed = seed
        self.rng = random.Random(seed)
        self.np_rng = np.random.default_rng(seed)
        self._init_state()

    def _init_state(self):
        self.step_count = 0
        self.total_reward = 0.0
        self.recycled_count = 0
        self.total_items = 0
        self.co2_total = 0.0
        self.co2_budget = 10.0

        self.facility_capacity = {
            0: self.rng.uniform(0.3, 0.9) * MAX_CAPACITY,
            1: self.rng.uniform(0.3, 0.9) * MAX_CAPACITY,
            2: self.rng.uniform(0.3, 0.9) * MAX_CAPACITY,
            3: self.rng.uniform(0.3, 0.9) * MAX_CAPACITY,
            4: self.rng.uniform(0.3, 0.9) * MAX_CAPACITY,
        }

        self.current_waste = self._generate_waste()
        self.recycling_rate = 0.0

    def _generate_waste(self) -> Dict[str, Any]:
        waste_type = self.rng.randint(0, 5)
        quantity = round(self.rng.uniform(5.0, 100.0), 2)
        zone = self.rng.randint(0, 2)
        time_of_day = self.rng.randint(0, 23)
        return {
            "waste_type": waste_type,
            "quantity": quantity,
            "zone": zone,
            "time_of_day": time_of_day,
        }

    def _get_observation(self) -> List[float]:
        w = self.current_waste
        caps = [
            self.facility_capacity[i] / MAX_CAPACITY for i in range(5)
        ]
        return [
            float(w["waste_type"]) / 5.0,
            float(w["quantity"]) / 100.0,
            float(w["zone"]) / 2.0,
            float(w["time_of_day"]) / 23.0,
            float(self.recycling_rate),
            caps[0], caps[1], caps[2], caps[3], caps[4],
            float(self.step_count) / MAX_STEPS,
            float(self.co2_budget) / 10.0,
        ]

    def _get_observation_dict(self) -> Dict[str, Any]:
        w = self.current_waste
        return {
            "waste_type": w["waste_type"],
            "waste_type_name": WASTE_TYPES[w["waste_type"]],
            "quantity": w["quantity"],
            "zone": w["zone"],
            "time_of_day": w["time_of_day"],
            "recycling_rate": round(self.recycling_rate, 4),
            "facility_capacity": {
                FACILITIES[i]: round(self.facility_capacity[i] / MAX_CAPACITY, 3)
                for i in range(5)
            },
            "step": self.step_count,
            "co2_budget_remaining": round(self.co2_budget, 3),
            "vector": [round(v, 4) for v in self._get_observation()],
        }

    def reset(self) -> Dict[str, Any]:
        self.rng = random.Random(self.seed)
        self.np_rng = np.random.default_rng(self.seed)
        self._init_state()
        return self._get_observation_dict()

    def step(self, action: int) -> Tuple[Dict, float, bool, Dict]:
        assert 0 <= action <= 4, f"Invalid action {action}. Must be 0-4."

        waste_type = self.current_waste["waste_type"]
        quantity = self.current_waste["quantity"]
        zone = self.current_waste["zone"]
        time_of_day = self.current_waste["time_of_day"]

        correct_facilities = CORRECT_FACILITY[waste_type]
        optimal = OPTIMAL_FACILITY[waste_type]
        is_correct = action in correct_facilities
        is_optimal = action == optimal

        distance_mult = ZONE_DISTANCE_MULTIPLIER[zone]
        peak_mult = 1.3 if 7 <= time_of_day <= 9 or 17 <= time_of_day <= 19 else 1.0
        base = BASE_COST[action]
        actual_cost = base * distance_mult * peak_mult * (quantity / 100.0)
        optimal_cost = BASE_COST[optimal] * distance_mult * peak_mult * (quantity / 100.0)
        max_cost = BASE_COST[2] * 1.9 * 1.3 * (quantity / 100.0)
        cost_efficiency = 1.0 - min(actual_cost / (max_cost + 1e-9), 1.0)

        co2 = CO2_EMISSION.get((waste_type, action), 0.7)
        co2_scaled = co2 * (quantity / 100.0)
        self.co2_total += co2_scaled
        self.co2_budget = max(0.0, self.co2_budget - co2_scaled)
        co2_score = 1.0 - min(co2 / 1.5, 1.0)

        cap_available = self.facility_capacity[action] / MAX_CAPACITY
        capacity_ok = cap_available > 0.05
        capacity_score = min(cap_available, 1.0)

        if is_correct and capacity_ok:
            self.facility_capacity[action] = max(
                0.0, self.facility_capacity[action] - quantity
            )

        if action == 0 or action == 1:
            self.recycled_count += 1
        self.total_items += 1
        self.recycling_rate = self.recycled_count / self.total_items

        correctness_score = 1.0 if is_optimal else (0.5 if is_correct else 0.0)
        if not capacity_ok:
            correctness_score *= 0.3

        if self.task_mode == "easy":
            reward = correctness_score
        elif self.task_mode == "medium":
            reward = 0.6 * correctness_score + 0.4 * cost_efficiency
        else:
            reward = (
                0.4 * correctness_score
                + 0.3 * cost_efficiency
                + 0.2 * co2_score
                + 0.1 * capacity_score
            )

        reward = float(np.clip(reward, 0.0, 1.0))

        self.step_count += 1
        self.total_reward += reward
        done = self.step_count >= MAX_STEPS

        info = {
            "waste_type": WASTE_TYPES[waste_type],
            "facility_chosen": FACILITIES[action],
            "is_correct": is_correct,
            "is_optimal": is_optimal,
            "cost": round(actual_cost, 4),
            "optimal_cost": round(optimal_cost, 4),
            "cost_efficiency": round(cost_efficiency, 4),
            "co2_emitted": round(co2_scaled, 4),
            "co2_score": round(co2_score, 4),
            "capacity_available": round(cap_available, 4),
            "recycling_rate": round(self.recycling_rate, 4),
            "task_mode": self.task_mode,
            "episode_reward_total": round(self.total_reward, 4),
        }

        if not done:
            self.current_waste = self._generate_waste()

        obs = self._get_observation_dict()
        return obs, reward, done, info

    def state(self) -> Dict[str, Any]:
        return {
            "task_mode": self.task_mode,
            "step": self.step_count,
            "max_steps": MAX_STEPS,
            "current_waste": {
                **self.current_waste,
                "waste_type_name": WASTE_TYPES[self.current_waste["waste_type"]],
            },
            "facility_capacity": {
                FACILITIES[i]: round(self.facility_capacity[i] / MAX_CAPACITY, 3)
                for i in range(5)
            },
            "recycling_rate": round(self.recycling_rate, 4),
            "co2_budget_remaining": round(self.co2_budget, 3),
            "co2_total_emitted": round(self.co2_total, 4),
            "episode_reward_total": round(self.total_reward, 4),
            "done": self.step_count >= MAX_STEPS,
        }

    def compute_grade(self) -> Dict[str, float]:
        avg_reward = self.total_reward / max(self.step_count, 1)
        recycling_score = self.recycling_rate
        co2_score = max(0.0, 1.0 - self.co2_total / 10.0)

        if self.task_mode == "easy":
            correctness_grader = float(np.clip(avg_reward, 0.0, 1.0))
            return {"score": correctness_grader, "grader": "correctness_grader"}

        elif self.task_mode == "medium":
            efficiency_grader = float(np.clip(
                0.6 * avg_reward + 0.4 * recycling_score, 0.0, 1.0
            ))
            return {"score": efficiency_grader, "grader": "efficiency_grader"}

        else:
            sustainability_grader = float(np.clip(
                0.4 * avg_reward + 0.3 * recycling_score + 0.3 * co2_score,
                0.0, 1.0,
            ))
            return {
                "score": sustainability_grader,
                "grader": "sustainability_grader",
                "recycling_rate": round(recycling_score, 4),
                "co2_score": round(co2_score, 4),
            }