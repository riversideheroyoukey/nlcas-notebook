from abc import ABC, abstractmethod
from notebook.models.ir import Expr

class BaseCodeGenerator(ABC):
    """Abstract base class for all Target Backend Code Generators.
    
    Transforms a semantically resolved AST into a target CAS command string.
    """
    
    @abstractmethod
    def generate(self, expr: Expr) -> str:
        """Generate target code from the given semantically resolved AST.
        
        Args:
            expr: The semantically resolved AST expression.
            
        Returns:
            The generated code string for the target backend.
        """
        pass
