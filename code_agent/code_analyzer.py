from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
import ast
import json
from pathlib import Path

# Load environment variables
load_dotenv()

class CodeAnalysisAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    
    def analyze_file(self, file_path: str) -> Dict[str, Any]:
        """Analyze a single Python file."""
        try:
            with open(file_path, 'r', encoding='utf-8') as file:
                content = file.read()
            
            # Parse the AST
            tree = ast.parse(content)
            
            # Extract basic information
            functions = []
            classes = []
            imports = []
            
            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    functions.append({
                        "name": node.name,
                        "args": [arg.arg for arg in node.args.args],
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.ClassDef):
                    classes.append({
                        "name": node.name,
                        "methods": [n.name for n in node.body if isinstance(n, ast.FunctionDef)],
                        "docstring": ast.get_docstring(node)
                    })
                elif isinstance(node, ast.Import):
                    imports.extend([n.name for n in node.names])
                elif isinstance(node, ast.ImportFrom):
                    imports.extend([f"{node.module}.{n.name}" for n in node.names])
            
            # Generate documentation
            doc_prompt = f"""Please analyze this Python code and provide:
            1. A high-level overview
            2. Documentation for each function and class
            3. Potential improvements or issues
            4. Test cases for key functions
            
            Code:
            {content}
            """
            
            documentation = self.llm.predict(doc_prompt)
            
            return {
                "file_path": file_path,
                "functions": functions,
                "classes": classes,
                "imports": imports,
                "documentation": documentation
            }
            
        except Exception as e:
            return {"error": str(e)}
    
    def analyze_directory(self, directory_path: str) -> Dict[str, Any]:
        """Analyze all Python files in a directory."""
        results = {}
        directory = Path(directory_path)
        
        for file_path in directory.rglob("*.py"):
            relative_path = str(file_path.relative_to(directory))
            results[relative_path] = self.analyze_file(str(file_path))
        
        # Generate project overview
        overview_prompt = f"""Based on the analysis of multiple Python files, provide:
        1. Project structure overview
        2. Key components and their relationships
        3. Main functionality
        4. Potential architectural improvements
        
        Analysis results:
        {json.dumps(results, indent=2)}
        """
        
        project_overview = self.llm.predict(overview_prompt)
        
        return {
            "project_overview": project_overview,
            "file_analyses": results
        }
    
    def generate_tests(self, file_path: str) -> Dict[str, Any]:
        """Generate test cases for a Python file."""
        analysis = self.analyze_file(file_path)
        
        test_prompt = f"""Based on the following code analysis, generate comprehensive test cases:
        1. Unit tests for each function
        2. Integration tests for class interactions
        3. Edge cases and error conditions
        
        Analysis:
        {json.dumps(analysis, indent=2)}
        """
        
        test_cases = self.llm.predict(test_prompt)
        
        return {
            "file_path": file_path,
            "test_cases": test_cases
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = CodeAnalysisAgent()
    
    # Example file analysis
    try:
        # Analyze a single file
        file_analysis = agent.analyze_file("path/to/your/file.py")
        print("File Analysis Results:")
        print(json.dumps(file_analysis, indent=2))
        
        # Generate tests
        test_cases = agent.generate_tests("path/to/your/file.py")
        print("\nGenerated Test Cases:")
        print(test_cases["test_cases"])
        
        # Analyze a directory
        directory_analysis = agent.analyze_directory("path/to/your/project")
        print("\nProject Analysis:")
        print(directory_analysis["project_overview"])
        
    except Exception as e:
        print(f"Error during code analysis: {str(e)}") 