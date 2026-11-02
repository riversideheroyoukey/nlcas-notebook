from __future__ import annotations
from enum import Enum
from typing import Literal, Optional
from datetime import datetime, timezone
import uuid
from pydantic import BaseModel, Field


class SessionMode(str, Enum):
    standard_symbolic = "standard_symbolic"
    second_quantization = "second_quantization"


class FallbackPolicy(str, Enum):
    ask_before_cross_backend = "ask_before_cross_backend"
    never = "never"


class LLMSupervisionMode(str, Enum):
    full = "full"
    on_demand = "on_demand"
    disabled = "disabled"


class LLMSupervisionConfig(BaseModel):
    mode: LLMSupervisionMode = LLMSupervisionMode.on_demand
    model_location: Literal["local", "api"] = "local"
    supervise_before_execution: bool = True
    supervise_after_execution: bool = True
    auto_execute_suggestions: bool = False


class Session(BaseModel):
    session_id: str = Field(default_factory=lambda: f"session_{uuid.uuid4().hex[:8]}")
    title: str
    mode: SessionMode
    primary_backend: str
    fallback_policy: FallbackPolicy = FallbackPolicy.ask_before_cross_backend
    show_cas_syntax_to_user: bool = False
    language: list[str] = Field(default_factory=lambda: ["ja", "en"])
    llm_supervision: LLMSupervisionConfig = Field(default_factory=LLMSupervisionConfig)
    created_at: datetime = Field(default_factory=lambda: datetime.now(timezone.utc))
    active_branch_id: str = "main"
    domains: list[str] = Field(default_factory=list)
    # Geometric coordinate system for vector calculus operations (Maxima sessions only)
    # "cartesian" | "spherical" | "cylindrical" | "custom" | None (= cartesian fallback)
    coordinate_system: Optional[str] = None
    # Custom coordinate system definition (set when coordinate_system == "custom")
    custom_coord_vars: Optional[tuple[str, str, str]] = None   # e.g. ("xi", "eta", "phi")
    custom_coord_lame: Optional[tuple[str, str, str]] = None   # e.g. ("sqrt(xi^2+eta^2)", ...)
