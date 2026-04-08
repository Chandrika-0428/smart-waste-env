---
title: smart-waste-env
emoji: ♻️
colorFrom: green
colorTo: blue
sdk: docker
pinned: false
---
# Smart Waste Management Environment

An OpenEnv-compatible reinforcement learning environment where an AI agent learns to route waste to the correct facility, optimizing for correctness, cost efficiency, and environmental sustainability.

---

## Project Description

Cities worldwide misroute 30–40% of waste, costing billions and generating unnecessary CO₂. This environment simulates a city waste management system where an agent must decide — for each incoming waste item — which of 5 facilities to send it to. The agent learns to balance correctness, cost, and environmental impact across 3 progressively harder tasks.

---

## Action Space

| Action | Facility | Accepts |
|--------|----------|---------|
| 0 | Recycling Plant | Plastic, Glass, Metal |
| 1 | Compost Facility | Organic waste |
| 2 | Landfill | Any (penalized) |
| 3 | Hazardous Facility | Hazardous waste only |
| 4 | Energy Recovery | General, Plastic |

**Type:** `Discrete(5)`

---

## Observation Space

**Type:** `Box(12,)` — normalized float vector

| Index | Feature | Range |
|-------|---------|-------|
| 0 | waste_type (0=plastic … 5=general) | [0, 1] |
| 1 | quantity (kg) | [0, 1] |
| 2 | zone (0=near, 2=far) | [0, 1] |
| 3 | time_of_day | [0, 1] |
| 4 | recycling_rate | [0, 1] |
| 5–9 | facility capacity (5 facilities) | [0, 1] |
| 10 | episode step progress | [0, 1] |
| 11 | CO₂ budget remaining | [0, 1] |

---

## Task Descriptions

### Easy — Waste Classification
Route each waste item to a correct (or optimal) facility.
**Reward:** correctness score only (0.0–1.0)

### Medium — Cost-Efficient Routing
Route waste correctly while minimizing transport and processing costs.
**Reward:** 60% correctness + 40% cost efficiency

### Hard — Sustainable Waste Optimization
Optimize across correctness, cost, CO₂ emissions, and facility capacity.
**Reward:** 40% correctness + 30% cost efficiency + 20% CO₂ score + 10% capacity score

---

## API Endpoints

| Method | Path | Description |
|--------|------|-------------|
| POST | `/reset` | Reset environment, returns initial observation |
| POST | `/step` | Take action, returns observation + reward + done + info |
| GET | `/state` | Get current environment state |
| GET | `/grade` | Get final grader score for current episode |
| GET | `/health` | Health check |

---

## Setup

### Local (without Docker)

```bash
pip install -r requirements.txt
uvicorn src.api:app --host 0.0.0.0 --port 8000
```

### Run Inference

```bash
# Default (local)
python inference.py

# With custom API URL
API_BASE_URL=http://localhost:8000 MODEL_NAME=my-agent python inference.py

# Against Hugging Face Space
API_BASE_URL=https://your-space.hf.space HF_TOKEN=hf_xxx python inference.py
```

---

## Docker

### Build

```bash
docker build -t smart-waste-env .
```

### Run

```bash
docker run -p 8000:8000 smart-waste-env
```

### Run Inference Against Docker Container

```bash
API_BASE_URL=http://localhost:8000 python inference.py
```

---

## Sample Output

```
Connecting to API: http://localhost:8000
API ready. Running inference with model=random-baseline

[START] task=easy model=random-baseline api=http://localhost:8000
[STEP] step=1 action=0 facility=recycling_plant waste=plastic correct=True reward=1.0000 done=False
[STEP] step=2 action=3 facility=hazardous_facility waste=organic correct=False reward=0.0000 done=False
[STEP] step=3 action=1 facility=compost_facility waste=organic correct=True reward=1.0000 done=False
...
[END] task=easy steps=20 total_reward=14.2300 avg_reward=0.7115 grade=0.7115 grader=correctness_grader

[START] task=medium model=random-baseline api=http://localhost:8000
...
[END] task=medium steps=20 total_reward=12.4500 avg_reward=0.6225 grade=0.6441 grader=efficiency_grader

[START] task=hard model=random-baseline api=http://localhost:8000
...
[END] task=hard steps=20 total_reward=11.8700 avg_reward=0.5935 grade=0.5812 grader=sustainability_grader

All tasks complete.
```

---

## OpenEnv Compliance

- ✅ `POST /reset` — returns typed `ObservationModel`
- ✅ `POST /step` — returns typed `StepResponse` (observation + reward ∈ [0,1] + done + info)
- ✅ `GET /state` — returns typed `StateResponse`
- ✅ Pydantic models for all request/response types
- ✅ `openenv.yaml` with id, name, endpoints, tasks, metadata
- ✅ 3 tasks with separate deterministic graders
- ✅ Rewards normalized to [0.0, 1.0]
- ✅ Dockerfile exposes port 8000
- ✅ `inference.py` at root, API-based, prints `[START]` `[STEP]` `[END]`

---

## Reward Design

| Component | Easy | Medium | Hard |
|-----------|------|--------|------|
| Correctness | 100% | 60% | 40% |
| Cost efficiency | — | 40% | 30% |
| CO₂ score | — | — | 20% |
| Capacity score | — | — | 10% |

All rewards are clipped to **[0.0, 1.0]** and are deterministic given the same seed.