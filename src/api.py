from fastapi import FastAPI
from pydantic import BaseModel
from src.waste_env import WasteManagementEnv

app = FastAPI()

env = WasteManagementEnv(task_mode="hard")


class ActionRequest(BaseModel):
    action: int


@app.get("/")
def home():
    return {"message": "Waste Management Env API running"}


@app.post("/reset")
def reset():
    state, _ = env.reset()
    return {"state": state.tolist()}


@app.post("/step")
def step(req: ActionRequest):
    state, reward, done, _, _ = env.step(req.action)
    return {
        "state": state.tolist(),
        "reward": reward,
        "done": done
    }