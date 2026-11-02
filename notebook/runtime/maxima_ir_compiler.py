"""Maxima canonical string ↔ IR AST converter.

Reuses the SharedParser for frontend parsing and ExprGenerator for backend-agnostic generation.
"""
from __future__ import annotations

from notebook.models.ir import Expr
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.expr_generator import ExprGenerator

class MaximaIRCompiler(SharedParser, ExprGenerator):
    """Handles bidirectional conversion between Maxima canonical strings and IR AST.
    
    Inherits SharedParser for robust, pure syntax parsing and ExprGenerator for string rendering.
    """
    pass
