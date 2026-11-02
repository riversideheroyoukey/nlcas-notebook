from __future__ import annotations
from typing import Optional
from datetime import datetime, timezone
from pydantic import BaseModel, Field


class Branch(BaseModel):
    branch_id: str
    label: str = ""
    root_step_id: Optional[str] = None
    head_step_id: Optional[str] = None
    forked_from_step_id: Optional[str] = None
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    primary_backend: str = "inherit"
