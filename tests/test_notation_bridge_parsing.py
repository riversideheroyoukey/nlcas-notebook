import pytest
from notebook.runtime.maxima_latex_converter import MaximaLatexConverter
from notebook.runtime.shared_parser import SharedParser
from notebook.runtime.maxima_semantic_resolver import MaximaSemanticResolver
from notebook.runtime.maxima_code_generator import MaximaCodeGenerator
from notebook.models.symbol_context import SymbolContext

def test_sum_parsing():
    converter = MaximaLatexConverter()
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    generator = MaximaCodeGenerator()
    ctx = SymbolContext()

    latex = r"\sum_{i} x_i"
    canonical = converter.to_canonical(latex)
    ast = parser.parse(canonical)
    ir = resolver.resolve(ast, ctx)
    assert ir.__class__.__name__ == "IndexSum"
    
    code = generator.generate(ir)
    assert "sum" in code

def test_delta_parsing():
    converter = MaximaLatexConverter()
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    generator = MaximaCodeGenerator()
    ctx = SymbolContext()

    latex = r"\delta_{i,j}"
    canonical = converter.to_canonical(latex)
    ast = parser.parse(canonical)
    ir = resolver.resolve(ast, ctx)
    assert ir.__class__.__name__ == "KroneckerDelta"

    code = generator.generate(ir)
    assert "kron_delta" in code

def test_integral_parsing():
    converter = MaximaLatexConverter()
    parser = SharedParser()
    resolver = MaximaSemanticResolver()
    generator = MaximaCodeGenerator()
    ctx = SymbolContext()

    latex = r"\int f(x) dx"
    canonical = converter.to_canonical(latex)
    ast = parser.parse(canonical)
    ir = resolver.resolve(ast, ctx)
    assert ir.__class__.__name__ == "Integral"
    
    code = generator.generate(ir)
    assert "integrate" in code

