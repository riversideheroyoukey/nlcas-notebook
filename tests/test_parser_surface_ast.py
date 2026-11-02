import pytest
from notebook.runtime.shared_parser import SharedParser
from notebook.models.ir import Symbol, Subscript, BracketExpr, Juxtaposition, FunctionCall, Superscript, Number

def test_shared_parser_raw_sum():
    parser = SharedParser()
    
    # \sum_{i} x_i
    ast = parser.parse(r"\sum_{i} x_i")
    
    assert isinstance(ast, Juxtaposition)
    assert isinstance(ast.left, Subscript)
    assert ast.left.base.name == "sum"

def test_shared_parser_raw_delta():
    parser = SharedParser()
    ast = parser.parse(r"\delta_{i, j}")
    assert isinstance(ast, Subscript)
    assert ast.base.name == "delta"

def test_shared_parser_raw_integral():
    parser = SharedParser()
    ast = parser.parse(r"\int f(x) dx")
    assert isinstance(ast, Juxtaposition)
    assert ast.right.name == "dx"

