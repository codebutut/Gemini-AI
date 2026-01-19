import ast
import io
import os
import shutil
import subprocess
import sys
import tempfile
import tokenize
from pathlib import Path
from typing import Any
from pydantic import BaseModel, Field
from . import tool, validate_args
from .file_tools import FilePathArgs, find_in_files

class RefactorArgs(BaseModel):
    filepath: str = Field(..., description="Path to Python file.")
    changes: list[dict[str, Any]] = Field(..., description="List of refactoring operations.")

class GenerateTestsArgs(BaseModel):
    filepath: str = Field(..., description="Path to Python file to generate tests for.")
    output_dir: str | None = Field(None, description="Directory to save test files.")

class DebugArgs(BaseModel):
    code: str = Field(..., description="Python code to debug.")
    breakpoints: list[int] | None = Field(None, description="Line numbers to break at.")

class ProfileArgs(BaseModel):
    code: str = Field(..., description="Python code to profile.")
    function_name: str | None = Field(None, description="Specific function to profile.")

class SearchCodebaseArgs(BaseModel):
    query: str = Field(..., description="Regex pattern to search for.")
    directory: str = Field(".", description="Root directory to search.")
    file_pattern: str | None = Field(None, description="Glob pattern for file filtering (e.g., '*.py').")
    case_sensitive: bool = Field(False, description="Case sensitive search.")

class ExecutePythonEnvArgs(BaseModel):
    code: str = Field(..., description="Python code to execute.")
    imports: list[str] | None = Field(None, description="List of modules to import before execution.")

class MermaidArgs(BaseModel):
    code: str = Field(..., description="Mermaid.js diagram code.")
    output_file: str = Field("diagram.mmd", description="Path to save the mermaid file.")

class DependencyGraphArgs(BaseModel):
    directory: str = Field(".", description="The directory to analyze.")
    recursive: bool = Field(True, description="Whether to search recursively.")

class CodeAnalyzer:
    """Static code analysis utilities."""

    @staticmethod
    def analyze_code(filepath: str) -> str:
        """Analyzes Python code for complexity, style, and potential issues."""
        try:
            with open(filepath, encoding="utf-8") as f:
                code = f.read()

            tree = ast.parse(code)
            analysis = {
                "functions": 0,
                "classes": 0,
                "imports": 0,
                "complex_functions": [],
                "issues": [],
            }

            for node in ast.walk(tree):
                if isinstance(node, ast.FunctionDef):
                    analysis["functions"] += 1
                    complexity = CodeAnalyzer._calculate_complexity(node)
                    if complexity > 10:
                        analysis["complex_functions"].append(f"  - {node.name}: complexity {complexity}")
                elif isinstance(node, ast.ClassDef):
                    analysis["classes"] += 1
                elif isinstance(node, (ast.Import, ast.ImportFrom)):
                    analysis["imports"] += 1

                if isinstance(node, ast.ExceptHandler) and node.type is None:
                    analysis["issues"].append(f"Bare except clause found (line {node.lineno})")
                elif isinstance(node, ast.Assert):
                    analysis["issues"].append(f"Assert statement found (line {node.lineno})")

            report = [
                f"File: {filepath}",
                f"Functions: {analysis['functions']}",
                f"Classes: {analysis['classes']}",
                f"Imports: {analysis['imports']}",
            ]
            if analysis["complex_functions"]:
                report.append("Complex functions (>10):")
                report.extend(analysis["complex_functions"])
            if analysis["issues"]:
                report.append("Potential issues:")
                report.extend(analysis["issues"])

            return "\n".join(report)
        except Exception as e:
            return f"Error analyzing code: {str(e)}"

    @staticmethod
    def _calculate_complexity(node: ast.AST) -> int:
        complexity = 1
        for child in ast.walk(node):
            if isinstance(child, (ast.If, ast.While, ast.For, ast.Try)):
                complexity += 1
            elif isinstance(child, ast.BoolOp):
                complexity += len(child.values) - 1
        return complexity

@tool
@validate_args(FilePathArgs)
def analyze_python_file(filepath: str) -> str:
    """
    Analyze Python code for complexity, style, and potential issues.

    Args:
        filepath: Path to Python file to analyze.

    Returns:
        str: Analysis report or error message.
    """
    return CodeAnalyzer.analyze_code(filepath)

@tool
@validate_args(RefactorArgs)
def refactor_code(filepath: str, changes: list[dict[str, Any]]) -> str:
    """
    Refactors code based on specified changes.

    Args:
        filepath: Path to Python file.
        changes: List of refactoring operations.

    Returns:
        str: Success message or error message.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File '{filepath}' not found."

        with open(path, encoding="utf-8") as f:
            content = f.read()

        backup_path = filepath + ".backup"
        shutil.copy2(filepath, backup_path)

        for change in changes:
            if change.get("type") == "rename":
                old_name = change.get("old_name")
                new_name = change.get("new_name")
                if not old_name or not new_name:
                    continue

                try:
                    tokens = list(tokenize.generate_tokens(io.StringIO(content).readline))
                except tokenize.TokenError:
                    return "Error: Could not tokenize file (syntax error?)"

                replacements = [t for t in tokens if t.type == tokenize.NAME and t.string == old_name]
                lines = content.splitlines(keepends=True)
                new_lines = []
                replacements_by_line = {}
                for t in replacements:
                    row = t.start[0] - 1
                    replacements_by_line.setdefault(row, []).append(t)

                for i, line in enumerate(lines):
                    if i in replacements_by_line:
                        line_repls = sorted(replacements_by_line[i], key=lambda t: t.start[1], reverse=True)
                        for t in line_repls:
                            line = line[: t.start[1]] + new_name + line[t.end[1] :]
                        new_lines.append(line)
                    else:
                        new_lines.append(line)
                content = "".join(new_lines)

        with open(path, "w", encoding="utf-8") as f:
            f.write(content)
        return f"Code refactored. Backup saved to {backup_path}"
    except Exception as e:
        return f"Error refactoring code: {str(e)}"

@tool
@validate_args(GenerateTestsArgs)
def generate_tests(filepath: str, output_dir: str | None = None) -> str:
    """
    Generates test stubs for a Python module.

    Args:
        filepath: Path to Python file to generate tests for.
        output_dir: Directory to save test files (default: tests/ in same directory).

    Returns:
        str: Success message or error message.
    """
    try:
        path = Path(filepath)
        if not path.exists():
            return f"Error: File '{filepath}' not found."

        with open(path, encoding="utf-8") as f:
            code = f.read()

        tree = ast.parse(code)
        test_code = [
            "import unittest",
            "import pytest",
            "",
            f"from {path.stem} import *",
            "",
            "class TestGenerated(unittest.TestCase):",
        ]

        for node in ast.walk(tree):
            if isinstance(node, ast.FunctionDef):
                test_code.extend(
                    [
                        f"\n    def test_{node.name}(self):",
                        f"        '''Auto-generated test for function {node.name}'''",
                        "        # TODO: Implement test logic",
                        "        self.skipTest('Test not implemented')",
                    ]
                )

        test_code.extend(["", "if __name__ == '__main__':", "    unittest.main()"])

        final_output_dir = Path(output_dir) if output_dir else path.parent / "tests"
        final_output_dir.mkdir(exist_ok=True, parents=True)

        test_file = final_output_dir / f"test_{path.name}"
        with open(test_file, "w", encoding="utf-8") as f:
            f.write("\n".join(test_code))
        return f"Test stubs generated: {test_file}"
    except Exception as e:
        return f"Error generating tests: {str(e)}"

@tool
@validate_args(DebugArgs)
def debug_python(code: str, breakpoints: list[int] | None = None) -> str:
    """
    Debugs Python code with breakpoint support.

    Args:
        code: Python code to debug.
        breakpoints: Line numbers to break at.

    Returns:
        str: Debugging output.
    """
    try:
        with tempfile.NamedTemporaryFile(mode="w", suffix="_debug.py", delete=False, encoding="utf-8") as tmp:
            debug_code = f"import sys\nimport traceback\n\n{code}"
            tmp.write(debug_code)
            tmp_path = tmp.name

        try:
            result = subprocess.run([sys.executable, tmp_path], capture_output=True, text=True, timeout=30)
            output = []
            if result.stdout:
                output.append(f"Output:\n{result.stdout}")
            if result.stderr:
                output.append(f"Errors:\n{result.stderr}")
            return "\n".join(output) if output else "(No output)"
        finally:
            if os.path.exists(tmp_path):
                os.remove(tmp_path)
    except subprocess.TimeoutExpired:
        return "Error: Debug session timed out (30s limit)."

@tool
@validate_args(ProfileArgs)
def profile_code(code: str, function_name: str | None = None) -> str:
    """
    Profiles Python code for performance analysis.

    Args:
        code: Python code to profile.
        function_name: Specific function to profile (optional).

    Returns:
        str: Profiling results.
    """
    try:
        profile_script = f"import cProfile, pstats, io\ndef run_code():\n{chr(10).join('    ' + l for l in code.splitlines())}\nif __name__ == '__main__':\n    profiler = cProfile.Profile()\n    profiler.enable()\n    try: run_code()\n    except Exception as e: print(f'Error: {{e}}')\n    profiler.disable()\n    s = io.StringIO()\n    ps = pstats.Stats(profiler, stream=s).sort_stats('cumulative')\n    ps.print_stats(20)\n    print(s.getvalue())"
        result = subprocess.run([sys.executable, "-c", profile_script], capture_output=True, text=True, timeout=60)
        output = []
        if result.stdout:
            output.append(f"Profiling Results:\n{result.stdout}")
        if result.stderr:
            output.append(f"Profiling Errors:\n{result.stderr}")
        return "\n".join(output) if output else "No profiling output"
    except subprocess.TimeoutExpired:
        return "Error: Profiling timed out (60s limit)."

@tool
@validate_args(SearchCodebaseArgs)
def search_codebase(
    query: str, directory: str = ".", file_pattern: str | None = None, case_sensitive: bool = False
) -> str:
    """
    Fast code search using ripgrep (rg) if available, falling back to python.

    Args:
        query: Regex pattern to search for.
        directory: Root directory.
        file_pattern: Glob pattern (e.g., '*.py').
        case_sensitive: Case sensitive search.

    Returns:
        str: Search results.
    """
    try:
        # Try running rg
        cmd = ["rg", "--line-number", "--no-heading", "--color=never"]
        if not case_sensitive:
            cmd.append("--ignore-case")
        if file_pattern:
            cmd.extend(["--glob", file_pattern])

        cmd.extend([query, directory])

        try:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=10)
            if result.returncode == 0:
                return f"Ripgrep Matches:\n{result.stdout}"[:10000]  # Limit output
            elif result.returncode == 1:
                return "No matches found (rg)."
            elif result.returncode == 2:
                # rg error, fall through to python fallback
                pass
        except FileNotFoundError:
            # rg not installed
            pass

        # Fallback to Python implementation
        return find_in_files(directory, query, file_pattern or "*")

    except Exception as e:
        return f"Error searching codebase: {str(e)}"

@tool
@validate_args(ExecutePythonEnvArgs)
def execute_python_with_env(code: str, imports: list[str] | None = None) -> str:
    """
    Executes Python code with pre-loaded imports.

    Args:
        code: Python code to execute.
        imports: List of modules to import before execution.

    Returns:
        str: Combined stdout and stderr or error message.
    """
    try:
        import_code = "".join(f"import {imp}\n" for imp in (imports or []))
        full_code = f"{import_code}\n{code}"
        result = subprocess.run([sys.executable, "-c", full_code], capture_output=True, text=True, timeout=30)
        output = []
        if result.stdout:
            output.append(f"Output:\n{result.stdout}")
        if result.stderr:
            output.append(f"Errors:\n{result.stderr}")
        return "\n".join(output) if output else "(No output)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."

@tool
@validate_args(MermaidArgs)
def render_mermaid(code: str, output_file: str = "diagram.mmd") -> str:
    """
    Saves Mermaid.js diagram code to a file.

    Args:
        code: Mermaid.js diagram code.
        output_file: Path to save the mermaid file.

    Returns:
        str: Success message.
    """
    try:
        path = Path(output_file)
        path.parent.mkdir(parents=True, exist_ok=True)
        with open(path, "w", encoding="utf-8") as f:
            f.write(code)
        return f"Mermaid diagram saved to '{output_file}'. You can view it using a Mermaid live editor or compatible viewer."
    except Exception as e:
        return f"Error saving Mermaid diagram: {str(e)}"

@tool
@validate_args(DependencyGraphArgs)
def get_dependency_graph(directory: str = ".", recursive: bool = True) -> str:
    """
    Analyze Python files in a directory to map out imports and dependencies between modules.

    Args:
        directory: The directory to analyze.
        recursive: Whether to search recursively.

    Returns:
        str: A summary of dependencies or error message.
    """
    try:
        dependencies = {}
        path = Path(directory)
        
        pattern = "**/*.py" if recursive else "*.py"
        for py_file in path.glob(pattern):
            try:
                with open(py_file, "r", encoding="utf-8", errors="ignore") as f:
                    tree = ast.parse(f.read())
                
                rel_path = py_file.relative_to(path)
                module_name = str(rel_path).replace(os.sep, ".").removesuffix(".py")
                
                file_deps = []
                for node in ast.walk(tree):
                    if isinstance(node, ast.Import):
                        for alias in node.names:
                            file_deps.append(alias.name)
                    elif isinstance(node, ast.ImportFrom):
                        if node.module:
                            file_deps.append(node.module)
                
                dependencies[module_name] = sorted(list(set(file_deps)))
            except Exception:
                continue
        
        if not dependencies:
            return "No Python files found or could not parse them."
            
        output = ["Dependency Graph:"]
        for mod, deps in dependencies.items():
            output.append(f"  {mod} -> {', '.join(deps)}")
            
        return "\n".join(output)
    except Exception as e:
        return f"Error generating dependency graph: {str(e)}"
