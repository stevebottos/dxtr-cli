"""
Repository Analyzer Submodule

Analyzes cloned repositories by parsing Python code with AST,
extracting functions/classes, and generating summaries using LLM.
"""

import ast
import hashlib
from pathlib import Path
from typing import Any
from ollama import chat

# Model for fast summarization
ANALYSIS_MODEL = "nemotron-mini"  # 4096 context window


def _get_analysis_cache_path(repo_path: str) -> Path:
    """
    Generate cache path for repository analysis.

    Args:
        repo_path: Path to the repository

    Returns:
        Path to cache file
    """
    # Create hash from repo path
    path_hash = hashlib.md5(repo_path.encode()).hexdigest()[:8]
    # Extract repo name from path (e.g., .dxtr/repos/owner/repo -> owner_repo)
    parts = Path(repo_path).parts
    if len(parts) >= 2:
        cache_name = f"analysis_{parts[-2]}_{parts[-1]}_{path_hash}.md"
    else:
        cache_name = f"analysis_{path_hash}.md"

    return Path(".dxtr") / cache_name


class PythonModuleAnalyzer(ast.NodeVisitor):
    """
    AST visitor that extracts structure and metadata from Python modules.
    """

    def __init__(self, source_code: str, file_path: str):
        """
        Initialize analyzer.

        Args:
            source_code: Python source code
            file_path: Path to the file being analyzed
        """
        self.source_code = source_code
        self.file_path = file_path
        self.lines = source_code.split('\n')

        # Extracted data
        self.module_docstring = None
        self.imports = []
        self.functions = []
        self.classes = []

    def visit_Module(self, node: ast.Module):
        """Extract module-level docstring."""
        self.module_docstring = ast.get_docstring(node)
        self.generic_visit(node)

    def visit_Import(self, node: ast.Import):
        """Track import statements."""
        for alias in node.names:
            self.imports.append({
                'type': 'import',
                'module': alias.name,
                'alias': alias.asname,
            })
        self.generic_visit(node)

    def visit_ImportFrom(self, node: ast.ImportFrom):
        """Track from...import statements."""
        module = node.module or ''
        for alias in node.names:
            self.imports.append({
                'type': 'from_import',
                'module': module,
                'name': alias.name,
                'alias': alias.asname,
            })
        self.generic_visit(node)

    def visit_FunctionDef(self, node: ast.FunctionDef):
        """Extract function definitions."""
        # Get function signature
        args = []
        for arg in node.args.args:
            arg_name = arg.arg
            arg_type = None
            if arg.annotation:
                arg_type = ast.unparse(arg.annotation)
            args.append({'name': arg_name, 'type': arg_type})

        # Get return type
        return_type = None
        if node.returns:
            return_type = ast.unparse(node.returns)

        # Get docstring
        docstring = ast.get_docstring(node)

        # Get source code for this function
        try:
            start_line = node.lineno - 1
            end_line = node.end_lineno
            source = '\n'.join(self.lines[start_line:end_line])
        except:
            source = None

        # Extract function calls within this function
        calls = self._extract_calls(node)

        self.functions.append({
            'name': node.name,
            'args': args,
            'return_type': return_type,
            'docstring': docstring,
            'source': source,
            'calls': calls,
            'line_number': node.lineno,
            'is_async': isinstance(node, ast.AsyncFunctionDef),
        })

        # Don't visit nested functions for now
        # self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef):
        """Handle async functions."""
        self.visit_FunctionDef(node)

    def visit_ClassDef(self, node: ast.ClassDef):
        """Extract class definitions."""
        # Get class docstring
        docstring = ast.get_docstring(node)

        # Get base classes
        bases = []
        for base in node.bases:
            bases.append(ast.unparse(base))

        # Extract methods
        methods = []
        for item in node.body:
            if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Similar to function extraction but for methods
                args = []
                for arg in item.args.args:
                    arg_name = arg.arg
                    arg_type = None
                    if arg.annotation:
                        arg_type = ast.unparse(arg.annotation)
                    args.append({'name': arg_name, 'type': arg_type})

                return_type = None
                if item.returns:
                    return_type = ast.unparse(item.returns)

                method_docstring = ast.get_docstring(item)

                # Get source code
                try:
                    start_line = item.lineno - 1
                    end_line = item.end_lineno
                    source = '\n'.join(self.lines[start_line:end_line])
                except:
                    source = None

                calls = self._extract_calls(item)

                methods.append({
                    'name': item.name,
                    'args': args,
                    'return_type': return_type,
                    'docstring': method_docstring,
                    'source': source,
                    'calls': calls,
                    'line_number': item.lineno,
                    'is_async': isinstance(item, ast.AsyncFunctionDef),
                })

        self.classes.append({
            'name': node.name,
            'bases': bases,
            'docstring': docstring,
            'methods': methods,
            'line_number': node.lineno,
        })

        # Don't visit nested classes
        # self.generic_visit(node)

    def _extract_calls(self, node: ast.AST) -> list[str]:
        """
        Extract function/method calls within a node.

        Args:
            node: AST node to analyze

        Returns:
            List of called function names
        """
        calls = []

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                # Try to get the function name
                if isinstance(child.func, ast.Name):
                    calls.append(child.func.id)
                elif isinstance(child.func, ast.Attribute):
                    # For method calls like obj.method()
                    calls.append(child.func.attr)

        return list(set(calls))  # Deduplicate


def analyze_python_file(file_path: Path) -> dict[str, Any] | None:
    """
    Analyze a single Python file.

    Args:
        file_path: Path to Python file

    Returns:
        Dictionary with analysis results or None if parsing failed
    """
    try:
        source_code = file_path.read_text(encoding='utf-8')
    except Exception as e:
        print(f"  [Error reading {file_path.name}: {e}]")
        return None

    try:
        tree = ast.parse(source_code)
    except SyntaxError as e:
        print(f"  [Syntax error in {file_path.name}: {e}]")
        return None

    analyzer = PythonModuleAnalyzer(source_code, str(file_path))
    analyzer.visit(tree)

    return {
        'file_path': str(file_path),
        'module_docstring': analyzer.module_docstring,
        'imports': analyzer.imports,
        'functions': analyzer.functions,
        'classes': analyzer.classes,
    }


def find_python_files(repo_path: Path, max_files: int = 100) -> list[Path]:
    """
    Find all Python files in a repository.

    Args:
        repo_path: Path to repository
        max_files: Maximum number of files to analyze

    Returns:
        List of Python file paths
    """
    python_files = []

    # Patterns to exclude
    exclude_patterns = [
        '*/test/*', '*/tests/*',
        '*/__pycache__/*',
        '*/venv/*', '*/env/*', '*/.venv/*',
        '*/node_modules/*',
        '*/.git/*',
        '*/dist/*', '*/build/*',
        '*/.pytest_cache/*',
    ]

    for py_file in repo_path.rglob('*.py'):
        # Check if file matches any exclude pattern
        if any(py_file.match(pattern) for pattern in exclude_patterns):
            continue

        python_files.append(py_file)

        if len(python_files) >= max_files:
            break

    return sorted(python_files)


def _summarize_function(func_data: dict[str, Any], context: str = "") -> str:
    """
    Generate LLM summary of a function's implementation.

    Args:
        func_data: Function metadata including source code
        context: Additional context (e.g., file purpose, class name)

    Returns:
        Summary string
    """
    # Skip if no source code
    if not func_data.get('source'):
        return "No source available"

    # Load system prompt for function analysis
    from ..agent import _load_system_prompt
    system_prompt = _load_system_prompt("function_analysis")

    # Build prompt for summarization
    prompt_parts = []

    if context:
        prompt_parts.append(f"Context: {context}\n")

    prompt_parts.append(f"Function: {func_data['name']}")

    if func_data.get('docstring'):
        prompt_parts.append(f"Docstring: {func_data['docstring']}")

    prompt_parts.append(f"\nSource code:\n```python\n{func_data['source']}\n```")

    prompt = "\n".join(prompt_parts)

    # Check token estimate (rough: chars / 4)
    # Account for system prompt (~150 tokens)
    estimated_tokens = (len(system_prompt) + len(prompt)) // 4
    if estimated_tokens > 3500:  # Leave room for response
        # Truncate source if too long
        max_source_chars = (3500 - len(system_prompt) // 4 - len(context) // 4 - 200) * 4
        truncated_source = func_data['source'][:max_source_chars] + "\n... (truncated)"
        prompt_parts[-1] = f"\nSource code:\n```python\n{truncated_source}\n```"
        prompt = "\n".join(prompt_parts)

    try:
        response = chat(
            model=ANALYSIS_MODEL,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": prompt}
            ],
            options={
                "temperature": 0.3,
                "num_ctx": 4096,
            }
        )

        if hasattr(response.message, 'content'):
            return response.message.content.strip()
        else:
            return "Summary generation failed"

    except Exception as e:
        return f"Error: {str(e)}"


def _format_function_signature(func_data: dict[str, Any]) -> str:
    """
    Format a function signature as a string.

    Args:
        func_data: Function metadata

    Returns:
        Formatted signature
    """
    args_str = ", ".join([
        f"{arg['name']}: {arg['type']}" if arg['type'] else arg['name']
        for arg in func_data['args']
    ])

    return_str = f" -> {func_data['return_type']}" if func_data['return_type'] else ""
    async_str = "async " if func_data.get('is_async') else ""

    return f"{async_str}def {func_data['name']}({args_str}){return_str}"


def analyze_repository(repo_path: str | Path, summarize: bool = True) -> dict[str, Any]:
    """
    Analyze an entire repository.

    Args:
        repo_path: Path to repository
        summarize: Whether to generate LLM summaries for functions

    Returns:
        Dictionary with repository analysis
    """
    repo_path = Path(repo_path)

    if not repo_path.exists():
        return {'error': f'Repository path does not exist: {repo_path}'}

    print(f"\n[Analyzing repository: {repo_path.name}]")

    # Find Python files
    python_files = find_python_files(repo_path)
    print(f"  [Found {len(python_files)} Python file(s)]")

    # Analyze each file
    file_analyses = []
    for py_file in python_files:
        rel_path = py_file.relative_to(repo_path)
        print(f"  [Parsing: {rel_path}]")

        analysis = analyze_python_file(py_file)
        if analysis:
            analysis['relative_path'] = str(rel_path)

            # Generate summaries for functions if requested
            if summarize and analysis['functions']:
                print(f"    [Summarizing {len(analysis['functions'])} function(s)...]")
                for func in analysis['functions']:
                    context = f"File: {rel_path}"
                    func['llm_summary'] = _summarize_function(func, context)

            # Generate summaries for class methods
            if summarize and analysis['classes']:
                for cls in analysis['classes']:
                    if cls['methods']:
                        print(f"    [Summarizing {len(cls['methods'])} method(s) in class {cls['name']}...]")
                        for method in cls['methods']:
                            context = f"File: {rel_path}, Class: {cls['name']}"
                            method['llm_summary'] = _summarize_function(method, context)

            file_analyses.append(analysis)

    return {
        'repo_path': str(repo_path),
        'repo_name': repo_path.name,
        'total_files': len(python_files),
        'analyzed_files': len(file_analyses),
        'files': file_analyses,
    }


def format_analysis_as_markdown(analysis: dict[str, Any]) -> str:
    """
    Format repository analysis as markdown.

    Args:
        analysis: Analysis dictionary from analyze_repository()

    Returns:
        Markdown-formatted string
    """
    lines = []

    lines.append(f"# Repository Analysis: {analysis['repo_name']}\n")
    lines.append(f"**Path:** `{analysis['repo_path']}`")
    lines.append(f"**Files analyzed:** {analysis['analyzed_files']} / {analysis['total_files']}\n")
    lines.append("---\n")

    for file_data in analysis['files']:
        rel_path = file_data['relative_path']
        lines.append(f"## File: `{rel_path}`\n")

        if file_data['module_docstring']:
            lines.append(f"**Module docstring:**\n```\n{file_data['module_docstring']}\n```\n")

        # Imports
        if file_data['imports']:
            lines.append(f"**Imports:** {len(file_data['imports'])}")
            import_list = []
            for imp in file_data['imports'][:10]:  # Show first 10
                if imp['type'] == 'import':
                    import_list.append(f"`{imp['module']}`")
                else:
                    import_list.append(f"`{imp['name']}` from `{imp['module']}`")
            lines.append(", ".join(import_list))
            if len(file_data['imports']) > 10:
                lines.append(f"... and {len(file_data['imports']) - 10} more")
            lines.append("")

        # Functions
        if file_data['functions']:
            lines.append(f"### Functions ({len(file_data['functions'])})\n")
            for func in file_data['functions']:
                lines.append(f"#### `{func['name']}` (line {func['line_number']})\n")
                lines.append(f"**Signature:** `{_format_function_signature(func)}`\n")

                if func.get('docstring'):
                    lines.append(f"**Docstring:**\n```\n{func['docstring']}\n```\n")

                if func.get('llm_summary'):
                    lines.append(f"**Summary:** {func['llm_summary']}\n")

                if func.get('calls'):
                    lines.append(f"**Calls:** {', '.join([f'`{c}`' for c in func['calls'][:10]])}\n")

        # Classes
        if file_data['classes']:
            lines.append(f"### Classes ({len(file_data['classes'])})\n")
            for cls in file_data['classes']:
                lines.append(f"#### Class: `{cls['name']}` (line {cls['line_number']})\n")

                if cls['bases']:
                    lines.append(f"**Inherits from:** {', '.join([f'`{b}`' for b in cls['bases']])}\n")

                if cls.get('docstring'):
                    lines.append(f"**Docstring:**\n```\n{cls['docstring']}\n```\n")

                if cls['methods']:
                    lines.append(f"**Methods ({len(cls['methods'])}):**\n")
                    for method in cls['methods']:
                        lines.append(f"- `{method['name']}` (line {method['line_number']})")
                        lines.append(f"  - Signature: `{_format_function_signature(method)}`")

                        if method.get('docstring'):
                            # Truncate long docstrings
                            doc = method['docstring']
                            if len(doc) > 200:
                                doc = doc[:200] + "..."
                            lines.append(f"  - Docstring: {doc}")

                        if method.get('llm_summary'):
                            lines.append(f"  - Summary: {method['llm_summary']}")

                        lines.append("")

        lines.append("\n---\n")

    return "\n".join(lines)
