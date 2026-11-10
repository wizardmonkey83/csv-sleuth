from pydantic import BaseModel, Field
from typing import Literal

from langchain.chat_models import init_chat_model
from langchain_core.messages import HumanMessage, SystemMessage

from .state import SleuthState
from .profile import render_profile

class Hypothesis(BaseModel):
    description: str = Field(description="One sentence: what might be true about the data")
    evidence: str = Field(description="What in the profile suggests this, citing the specific numbers or values.")
    check: str = Field(description="How to test it with pandas, described in plain words (not code).")
    priority: Literal["high", "medium", "low"] = Field(description="How strongly the profile points to this.")

class Plan(BaseModel):
    hypotheses: list[Hypothesis] = Field(description="4 to 8 hypotheses, most suspicious first.", min_length=4, max_length=8)


OPENAI_MODEL = "openai:gpt-6-luna"
SYSTEM_PROMPT = ""

model = init_chat_model(model=OPENAI_MODEL, temperature=0)
planner = model.with_structured_output(Plan)

def plan_node(state: SleuthState) -> dict:
    profile_text = render_profile(state["profile"])
    messages = [
        SystemMessage(content=SYSTEM_PROMPT),
        HumanMessage(content=f"This is the rendered profile of the dataset: \n\n{profile_text}")
    ]
    plan = planner.invoke(messages)
    plan_dict = plan.model_dump()

    hypotheses = []
    for i, hypothesis in enumerate(plan_dict["hypotheses"], start=1):
        hypothesis["id"] = f"h{i}"
        hypotheses.append(hypothesis)

    return {"hypotheses": hypotheses}

