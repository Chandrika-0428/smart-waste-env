# Smart Waste Management Environment (OpenEnv)

## Problem
Waste management systems are inefficient due to incorrect routing of waste, leading to higher costs and environmental damage.

## Solution
We built a reinforcement learning environment where an AI agent learns to optimally route waste based on:
- waste type
- cost
- environmental impact
- location

## Tasks
1. Easy: Waste classification
2. Medium: Cost-aware routing
3. Hard: Multi-objective optimization (cost + environment)

## Features
- Realistic waste types and facilities
- Cost modeling with distance (zones)
- Environmental scoring
- Multi-task difficulty levels
- Deterministic grading system

## Tech Stack
- Python
- Gymnasium
- FastAPI
- Docker

## How to Run
```bash
docker build -t waste-env .
docker run -p 8000:8000 waste-env