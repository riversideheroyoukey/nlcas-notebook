# Re-export shim — implementation lives in maxima_ir_compiler.py
from notebook.runtime.maxima_ir_compiler import MaximaIRCompiler

__all__ = ["MaximaIRCompiler"]
