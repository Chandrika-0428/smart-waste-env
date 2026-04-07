# ==============================================
# Smart Waste Management Environment
# ==============================================
# STATE:  waste_type, quantity, zone,
#         facility_capacity[5], time_of_day,
#         recycling_rate
#
# ACTIONS:
#   0 = Recycling plant
#   1 = Compost facility
#   2 = Landfill
#   3 = Hazardous facility
#   4 = Energy recovery
#
# EPISODE: 20 waste items per episode
# REWARD:  shaped (correctness + cost + CO2)
# Reward = 0.5 correctness + 0.3 cost + 0.2 environment
# ==============================================

import gymnasium as gym
from gymnasium import spaces
import numpy as np
import random


class WasteManagementEnv(gym.Env):

    def __init__(self, task_mode="easy"):
        super(WasteManagementEnv, self).__init__()

        # Waste types
        self.waste_types = ["plastic", "organic", "glass", "metal", "hazardous", "general"]

        # Actions (5 facilities)
        self.action_space = spaces.Discrete(5)

        # Observation space
        self.observation_space = spaces.Box(low=0, high=100, shape=(3,), dtype=np.float32)

        # Task mode
        self.task_mode = task_mode

        # Episode control
        self.max_steps = 20
        self.current_step = 0

        # Facility costs
        self.facility_costs = [10, 8, 5, 20, 12]

        # Capacity (not used yet, but useful for later)
        self.capacity = [500, 300, 1000, 100, 400]

        # Zone multipliers
        self.zone_multiplier = {
            "A": 1.0,
            "B": 1.5,
            "C": 2.0
        }

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)

        self.current_step = 0

        # Random state
        self.waste_type = random.choice(self.waste_types)
        self.quantity = random.randint(1, 10)
        self.cost = random.uniform(1, 10)
        self.zone = random.choice(["A", "B", "C"])
        self.time = random.choice(["peak", "off"])


        state = np.array([
            self.quantity,
            self.cost,
            self.waste_types.index(self.waste_type)
        ], dtype=np.float32)

        return state, {}

    def step(self, action):
        self.current_step += 1

        # -------------------------
        # CORRECTNESS LOGIC
        # -------------------------
        correct_map = {
            "plastic": 0,
            "organic": 1,
            "glass": 0,
            "metal": 0,
            "hazardous": 3,
            "general": 2
        }

        correct_action = correct_map[self.waste_type]
        correctness = 1 if action == correct_action else 0

        # -------------------------
        # COST LOGIC
        # -------------------------
        base_cost = self.facility_costs[action]
        zone_factor = self.zone_multiplier[self.zone]

        total_cost = base_cost * zone_factor * self.quantity
                # random disruption
        if random.random() < 0.1:
            total_cost *= 1.5

        # time effect
        if self.time == "peak":
            total_cost *= 1.2
            max_cost = 50
        cost_eff = max(0, 1 - (total_cost / max_cost))

        # -------------------------
        # ENVIRONMENTAL SCORE
        # -------------------------
        env_scores = [1.0, 0.9, 0.2, 0.0, 0.6]
        env_score = env_scores[action]

                # capacity penalty
        if self.capacity[action] < self.quantity:
            correctness = 0
            env_score *= 0.5

        # -------------------------
        # REWARD (BY TASK MODE)
        # -------------------------
        if self.task_mode == "easy":
            reward = correctness

        elif self.task_mode == "medium":
            reward = 0.6 * correctness + 0.4 * cost_eff

        elif self.task_mode == "hard":
            reward = 0.5 * correctness + 0.3 * cost_eff + 0.2 * env_score

        else:
            reward = correctness  # fallback

        # -------------------------
        # NEXT STATE
        # -------------------------
        self.waste_type = random.choice(self.waste_types)
        self.quantity = random.randint(1, 10)
        self.cost = random.uniform(1, 10)
        self.zone = random.choice(["A", "B", "C"])

        state = np.array([
            self.quantity,
            self.cost,
            self.waste_types.index(self.waste_type)
        ], dtype=np.float32)

        done = self.current_step >= self.max_steps

        return state, reward, done, False, {}