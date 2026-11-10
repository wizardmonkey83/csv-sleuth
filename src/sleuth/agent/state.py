from __future__ import annotations
 
from typing import TypedDict
 
from .profile import DataProfile
 
class SleuthState(TypedDict, total=False):
    csv_path: str
    profile: DataProfile
    hypotheses: list[dict]