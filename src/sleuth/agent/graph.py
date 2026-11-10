from dotenv import load_dotenv
load_dotenv()

import os

from langgraph.graph import StateGraph, START, END

from .plan import plan_node
from .profile import profile_node
from .state import SleuthState

CSV_PATH = os.getenv("CSV_PATH")
if CSV_PATH is None:
    raise ValueError(("CSV_PATH is not set in .env"))

def profile(state: SleuthState):
    profile = profile_node(state=state)
    return {"profile": profile}

def plan(state: SleuthState):
    profile = state["profile"]
    hypotheses = plan_node(profile=profile)
    return {"hypotheses": hypotheses}






graph = StateGraph(SleuthState)

graph.add_node("profile", profile)
graph.add_node("plan", plan)

graph.add_edge(START, "profile")
graph.add_edge("profile", "plan")

if __name__ == "__main__":
    graph.invoke({"csv_path": CSV_PATH})