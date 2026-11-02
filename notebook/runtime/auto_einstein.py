from typing import List, Set, Dict, Optional
from notebook.models.ir import Expr, Mul, Add, Pow, IndexedSymbol, IndexSymbol, IndexSum, FunctionCall, Equation, Derivative, Integral, Commutator

def auto_einstein_sum(expr: Expr) -> Expr:
    """Wraps multiplicative terms containing matching upper and lower indices with IndexSum."""
    
    if isinstance(expr, Mul):
        # Gather all index symbols inside this Mul term
        up_indices = set()
        down_indices = set()
        
        def collect_indices(e: Expr):
            if isinstance(e, IndexedSymbol):
                for idx in e.indices:
                    if isinstance(idx, IndexSymbol):
                        if getattr(idx, "is_up", False):
                            up_indices.add(idx.name)
                        else:
                            down_indices.add(idx.name)
            elif isinstance(e, Mul):
                for arg in e.args:
                    collect_indices(arg)
            elif isinstance(e, Pow):
                collect_indices(e.base)
        
        collect_indices(expr)
        
        # Find intersecting indices (dummy indices)
        dummy = up_indices.intersection(down_indices)
        
        # Recursively process args first
        new_args = [auto_einstein_sum(arg) for arg in expr.args]
        new_mul = Mul(args=new_args)
        
        if dummy:
            # Wrap with IndexSum
            dummy_symbols = [IndexSymbol(name=name) for name in sorted(dummy)]
            return IndexSum(body=new_mul, dummy_indices=dummy_symbols)
        return new_mul

    elif isinstance(expr, Add):
        return Add(args=[auto_einstein_sum(arg) for arg in expr.args])
    elif isinstance(expr, Pow):
        return Pow(base=auto_einstein_sum(expr.base), exponent=auto_einstein_sum(expr.exponent))
    elif isinstance(expr, FunctionCall):
        return FunctionCall(func=auto_einstein_sum(expr.func), args=[auto_einstein_sum(arg) for arg in expr.args])
    elif isinstance(expr, Equation):
        return Equation(lhs=auto_einstein_sum(expr.lhs), rhs=auto_einstein_sum(expr.rhs), op=expr.op)
    elif isinstance(expr, Derivative):
        return Derivative(expr=auto_einstein_sum(expr.expr), vars=[(auto_einstein_sum(v), o) for v, o in expr.vars], is_partial=expr.is_partial)
    elif isinstance(expr, Integral):
        return expr.model_copy(update={
            "expr": auto_einstein_sum(expr.expr),
            "var": auto_einstein_sum(expr.var),
            "lower": auto_einstein_sum(expr.lower) if expr.lower else None,
            "upper": auto_einstein_sum(expr.upper) if expr.upper else None,
        })
    elif isinstance(expr, Commutator):
        return Commutator(a=auto_einstein_sum(expr.a), b=auto_einstein_sum(expr.b))
    
    return expr
