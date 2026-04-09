"""
Smart Waste Management Environment — FastAPI Server
"""
from __future__ import annotations

from typing import Any, Dict, Optional

from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, Field

from src.env import WasteManagementEnv

app = FastAPI(title="Smart Waste Management Environment", version="1.0.0")

# One env instance per task mode, keyed by task_mode string
_envs: Dict[str, WasteManagementEnv] = {}


def _get_env(task_mode: str) -> WasteManagementEnv:
    if task_mode not in ("easy", "medium", "hard"):
        raise HTTPException(status_code=422, detail=f"Unknown task_mode: {task_mode}")
    return _envs.get(task_mode)


# ---------- Request / Response Models ----------

class ResetRequest(BaseModel):
    task_mode: Optional[str] = Field(default="easy")
    seed: Optional[int] = Field(default=42)


class StepRequest(BaseModel):
    action: int
    task_mode: Optional[str] = Field(default="easy")


# ---------- Endpoints ----------

@app.post("/reset")
def reset(body: ResetRequest = None) -> Dict[str, Any]:
    """Reset the environment. Accepts empty body — defaults to easy/seed=42."""
    if body is None:
        body = ResetRequest()

    task_mode = body.task_mode or "easy"
    seed = body.seed if body.seed is not None else 42

    env = WasteManagementEnv(task_mode=task_mode, seed=seed)
    _envs[task_mode] = env

    obs = env.reset()
    return {
        "observation": obs,
        "reward": 0.0,
        "done": False,
        "info": {},
    }


@app.post("/step")
def step(body: StepRequest) -> Dict[str, Any]:
    task_mode = body.task_mode or "easy"
    env = _get_env(task_mode)
    if env is None:
        # Auto-init if not reset yet
        env = WasteManagementEnv(task_mode=task_mode, seed=42)
        _envs[task_mode] = env

    result = env.step(body.action)
    return result


@app.get("/state")
def state(task_mode: str = "easy") -> Dict[str, Any]:
    env = _get_env(task_mode)
    if env is None:
        raise HTTPException(status_code=400, detail="Environment not initialized. Call /reset first.")
    return env.state()


@app.get("/grade")
def grade(task_mode: str = "easy") -> Dict[str, Any]:
    env = _get_env(task_mode)
    if env is None:
        return {"score": 0.0, "grader": "correctness_grader", "steps": 0}
    return env.grade()


@app.get("/health")
def health() -> Dict[str, str]:
    return {"status": "ok"}


def main():
    import uvicorn
    uvicorn.run("src.api:app", host="0.0.0.0", port=8000, reload=False)