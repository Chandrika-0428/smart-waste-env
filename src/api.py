from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel, Field
from typing import Any, Dict, List, Optional
import os

from src.waste_env import WasteManagementEnv

app = FastAPI(
    title="Smart Waste Management Environment API",
    description="OpenEnv-compatible RL environment for waste routing optimization.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

_envs: Dict[str, WasteManagementEnv] = {}

VALID_TASK_MODES = {"easy", "medium", "hard"}


def _get_env(task_mode: str) -> WasteManagementEnv:
    if task_mode not in _envs:
        _envs[task_mode] = WasteManagementEnv(task_mode=task_mode)
    return _envs[task_mode]


class ResetRequest(BaseModel):
    task_mode: str = Field(default="easy", description="Task difficulty: easy, medium, hard")
    seed: int = Field(default=42, description="Random seed for reproducibility")


class ActionModel(BaseModel):
    action: int = Field(
        ...,
        ge=0,
        le=4,
        description=(
            "Facility to route waste to: "
            "0=Recycling Plant, 1=Compost Facility, "
            "2=Landfill, 3=Hazardous Facility, 4=Energy Recovery"
        ),
    )
    task_mode: str = Field(default="easy", description="Task difficulty: easy, medium, hard")


class FacilityCapacity(BaseModel):
    recycling_plant: float
    compost_facility: float
    landfill: float
    hazardous_facility: float
    energy_recovery: float


class ObservationModel(BaseModel):
    waste_type: int
    waste_type_name: str
    quantity: float
    zone: int
    time_of_day: int
    recycling_rate: float
    facility_capacity: Dict[str, float]
    step: int
    co2_budget_remaining: float
    vector: List[float]


class StepInfoModel(BaseModel):
    waste_type: str
    facility_chosen: str
    is_correct: bool
    is_optimal: bool
    cost: float
    optimal_cost: float
    cost_efficiency: float
    co2_emitted: float
    co2_score: float
    capacity_available: float
    recycling_rate: float
    task_mode: str
    episode_reward_total: float


class StepResponse(BaseModel):
    observation: ObservationModel
    reward: float = Field(..., ge=0.0, le=1.0)
    done: bool
    info: StepInfoModel


class StateResponse(BaseModel):
    task_mode: str
    step: int
    max_steps: int
    current_waste: Dict[str, Any]
    facility_capacity: Dict[str, float]
    recycling_rate: float
    co2_budget_remaining: float
    co2_total_emitted: float
    episode_reward_total: float
    done: bool


class GradeResponse(BaseModel):
    score: float
    grader: str
    recycling_rate: Optional[float] = None
    co2_score: Optional[float] = None


@app.get("/", tags=["health"])
def root():
    return {
        "status": "ok",
        "name": "Smart Waste Management Environment",
        "version": "1.0.0",
        "endpoints": ["/reset", "/step", "/state", "/grade", "/health"],
    }


@app.get("/health", tags=["health"])
def health():
    return {"status": "healthy"}


@app.post("/reset", response_model=ObservationModel, tags=["environment"])
def reset(request: ResetRequest):
    if request.task_mode not in VALID_TASK_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"task_mode must be one of {VALID_TASK_MODES}",
        )
    env = WasteManagementEnv(task_mode=request.task_mode, seed=request.seed)
    _envs[request.task_mode] = env
    obs = env.reset()
    return ObservationModel(**obs)


@app.post("/step", response_model=StepResponse, tags=["environment"])
def step(request: ActionModel):
    if request.task_mode not in VALID_TASK_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"task_mode must be one of {VALID_TASK_MODES}",
        )
    env = _get_env(request.task_mode)
    if env.step_count >= 20:
        raise HTTPException(
            status_code=400,
            detail="Episode is done. Call /reset to start a new episode.",
        )
    obs, reward, done, info = env.step(request.action)
    return StepResponse(
        observation=ObservationModel(**obs),
        reward=reward,
        done=done,
        info=StepInfoModel(**info),
    )


@app.get("/state", response_model=StateResponse, tags=["environment"])
def state(task_mode: str = "easy"):
    if task_mode not in VALID_TASK_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"task_mode must be one of {VALID_TASK_MODES}",
        )
    env = _get_env(task_mode)
    return StateResponse(**env.state())


@app.get("/grade", response_model=GradeResponse, tags=["environment"])
def grade(task_mode: str = "easy"):
    if task_mode not in VALID_TASK_MODES:
        raise HTTPException(
            status_code=422,
            detail=f"task_mode must be one of {VALID_TASK_MODES}",
        )
    env = _get_env(task_mode)
    result = env.compute_grade()
    return GradeResponse(**result)

@app.get("/tasks", tags=["environment"])
def list_tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "name": "Waste Classification",
                "difficulty": "easy",
                "max_steps": 20,
                "reward_range": [0.0, 1.0],
                "grader": "correctness_grader"
            },
            {
                "id": "medium",
                "name": "Cost-Efficient Routing",
                "difficulty": "medium",
                "max_steps": 20,
                "reward_range": [0.0, 1.0],
                "grader": "efficiency_grader"
            },
            {
                "id": "hard",
                "name": "Sustainable Waste Optimization",
                "difficulty": "hard",
                "max_steps": 20,
                "reward_range": [0.0, 1.0],
                "grader": "sustainability_grader"
            }
        ]
    }

@app.get("/")
def root():
    return {"message": "Smart Waste Env is running"}