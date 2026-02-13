"""
ds-sandbox security scanner

Static code security analysis for detecting dangerous patterns
in Python code before execution.
"""

import ast
import re
from typing import Dict, List, Optional, Tuple
from dataclasses import dataclass

from ..types import CodeScanResult, CodeIssue


# Risk weights for different pattern categories
PATTERN_WEIGHTS = {
    "file_write": 0.4,
    "network": 0.3,
    "subprocess": 0.5,
    "dynamic_exec": 0.6,
    "system": 0.5,
}

# Severity mapping
SEVERITY_MAP = {
    "file_write": "high",
    "network": "medium",
    "subprocess": "high",
    "dynamic_exec": "critical",
    "system": "high",
}

# Module to pattern type mapping
MODULE_PATTERNS: Dict[str, str] = {
    "os": "file_write",
    "shutil": "file_write",
    "socket": "network",
    "urllib": "network",
    "requests": "network",
    "http": "network",
    "http.client": "network",
    "subprocess": "subprocess",
    "pdb": "system",
    "sys": "system",
}

# Dangerous functions by category
DANGEROUS_FUNCTIONS: Dict[str, List[str]] = {
    "file_write": ["remove", "rmdir", "unlink", "rename", "mkdir", "makedirs"],
    "network": ["socket", "create_connection"],
    "subprocess": ["Popen", "call", "run", "check_call", "check_output"],
    "dynamic_exec": ["exec", "eval", "compile", "__import__", "open"],
    "system": ["system", "exit", "exitfunc", "excepthook"],
}

# Allowed functions (whitelist) that are safe to use
SAFE_FUNCTIONS = {
    "print", "len", "str", "int", "float", "bool", "list", "dict", "set",
    "tuple", "range", "enumerate", "zip", "map", "filter", "sorted",
    "re", "json", "math", "random", "datetime", "collections", "itertools",
    "os.path", "os.getcwd", "os.path.join", "os.path.exists", "os.path.isfile",
    "os.path.isdir", "os.path.dirname", "os.path.basename", "os.path.splitext",
}


@dataclass
class PatternMatch:
    """Represents a pattern match found in code."""
    line: int
    pattern_type: str
    match_text: str
    severity: str
    weight: float
    function: Optional[str] = None
    module: Optional[str] = None


class CodeScanner:
    """
    Static code security scanner.

    Performs static analysis on Python code to detect dangerous patterns
    and calculate risk scores without executing the code.
    """

    # Regex patterns for dangerous operations
    DANGEROUS_PATTERNS = {
        "file_write": r"\b(os\.remove|os\.rmdir|os\.unlink|shutil\.rmtree|shutil\.move)\s*\(",
        "network": r"\b(socket\.|urllib\.|requests\.|http\.client\.)\s*\(",
        "subprocess": r"\b(subprocess\.(Popen|call|run|check_call|check_output))\s*\(",
        "dynamic_exec": r"\b(exec|eval|compile|__import__)\s*\(",
        "system": r"\b(os\.system|sys\.exit|pdb\.set_trace)\s*\(",
    }

    def __init__(self):
        """Initialize the code scanner."""
        self._compile_patterns()

    def _compile_patterns(self) -> None:
        """Compile regex patterns for efficient matching."""
        self._compiled_patterns: Dict[str, re.Pattern] = {}
        for name, pattern in self.DANGEROUS_PATTERNS.items():
            self._compiled_patterns[name] = re.compile(pattern)

    def scan(self, code: str) -> CodeScanResult:
        """
        Scan code for security issues.

        Args:
            code: Python code string to scan

        Returns:
            CodeScanResult with issues and risk score
        """
        issues: List[CodeIssue] = []
        total_risk = 0.0

        # Run pattern matching
        pattern_issues = self._scan_patterns(code)
        issues.extend(pattern_issues)

        # Run AST analysis
        ast_issues = self._scan_ast(code)
        issues.extend(ast_issues)

        # Remove duplicates based on line and type
        issues = self._deduplicate_issues(issues)

        # Calculate total risk score
        total_risk = self._calculate_risk_score(issues)

        # Determine if code is safe
        is_safe = total_risk < 0.3

        # Determine recommended backend
        recommended_backend = self._get_recommended_backend(total_risk, issues)

        return CodeScanResult(
            is_safe=is_safe,
            risk_score=total_risk,
            issues=issues,
            recommended_backend=recommended_backend,
        )

    def _scan_patterns(self, code: str) -> List[CodeIssue]:
        """Scan code using regex patterns."""
        issues: List[CodeIssue] = []

        for pattern_type, compiled in self._compiled_patterns.items():
            for match in compiled.finditer(code):
                line_num = code[:match.start()].count('\n') + 1
                match_text = match.group().strip()

                # Extract function/module info
                function, module = self._extract_function_info(match_text)

                # Create issue
                issue = CodeIssue(
                    type=pattern_type,
                    line=line_num,
                    severity=SEVERITY_MAP.get(pattern_type, "medium"),
                    weight=PATTERN_WEIGHTS.get(pattern_type, 0.3),
                    function=function,
                    module=module,
                )
                issues.append(issue)

        return issues

    def _scan_ast(self, code: str) -> List[CodeIssue]:
        """Scan code using AST analysis for additional checks."""
        issues: List[CodeIssue] = []
        try:
            tree = ast.parse(code)
            self._analyze_ast(tree, code, issues)
        except SyntaxError:
            # Syntax errors will be caught during execution
            pass
        return issues

    def _analyze_ast(
        self,
        node: ast.AST,
        source_code: str,
        issues: List[CodeIssue]
    ) -> None:
        """Recursively analyze AST nodes for dangerous patterns."""
        lines = source_code.split('\n')

        for child in ast.walk(node):
            if isinstance(child, ast.Call):
                self._check_call(child, lines, issues)

            elif isinstance(child, ast.Import):
                for alias in child.names:
                    if alias.name in MODULE_PATTERNS:
                        pattern_type = MODULE_PATTERNS[alias.name]
                        line_num = child.lineno
                        issue = CodeIssue(
                            type=pattern_type,
                            line=line_num,
                            severity=SEVERITY_MAP.get(pattern_type, "medium"),
                            weight=PATTERN_WEIGHTS.get(pattern_type, 0.3),
                            module=alias.name,
                        )
                        issues.append(issue)

            elif isinstance(child, ast.ImportFrom):
                if child.module and child.module.split('.')[0] in MODULE_PATTERNS:
                    pattern_type = MODULE_PATTERNS[child.module.split('.')[0]]
                    line_num = child.lineno
                    issue = CodeIssue(
                        type=pattern_type,
                        line=line_num,
                        severity=SEVERITY_MAP.get(pattern_type, "medium"),
                        weight=PATTERN_WEIGHTS.get(pattern_type, 0.3),
                        module=child.module,
                    )
                    issues.append(issue)

    def _check_call(
        self,
        call: ast.Call,
        lines: List[str],
        issues: List[CodeIssue]
    ) -> None:
        """Check function calls for dangerous patterns."""
        line_num = getattr(call, 'lineno', 1)

        # Check direct function calls
        if isinstance(call.func, ast.Name):
            func_name = call.func.id

            # Check for dangerous functions
            for category, funcs in DANGEROUS_FUNCTIONS.items():
                if func_name in funcs:
                    issue = CodeIssue(
                        type=category,
                        line=line_num,
                        severity=SEVERITY_MAP.get(category, "medium"),
                        weight=PATTERN_WEIGHTS.get(category, 0.3),
                        function=func_name,
                    )
                    issues.append(issue)

        # Check method calls
        elif isinstance(call.func, ast.Attribute):
            method_name = call.func.attr

            # Check for dynamic code execution
            if method_name in ("exec", "eval", "compile"):
                issue = CodeIssue(
                    type="dynamic_exec",
                    line=line_num,
                    severity="critical",
                    weight=0.6,
                    function=method_name,
                )
                issues.append(issue)

            # Check for file operations
            if method_name in ("read", "write", "open"):
                if isinstance(call.func.value, ast.Name):
                    # Check if it's a direct file.open() call
                    if call.func.value.id == "open":
                        issue = CodeIssue(
                            type="dynamic_exec",
                            line=line_num,
                            severity="high",
                            weight=0.4,
                            function="open",
                        )
                        issues.append(issue)

    def _extract_function_info(self, match_text: str) -> Tuple[Optional[str], Optional[str]]:
        """Extract function and module information from match text."""
        if '(' in match_text:
            func_part = match_text.rsplit('(', 1)[0]
            if '.' in func_part:
                parts = func_part.split('.')
                module = parts[0] if len(parts) > 1 else None
                function = parts[-1]
                return function, module
            else:
                return func_part, None
        return None, None

    def _deduplicate_issues(self, issues: List[CodeIssue]) -> List[CodeIssue]:
        """Remove duplicate issues based on line and type."""
        seen: set = set()
        unique: List[CodeIssue] = []

        for issue in issues:
            key = (issue.line, issue.type)
            if key not in seen:
                seen.add(key)
                unique.append(issue)

        return unique

    def _calculate_risk_score(self, issues: List[CodeIssue]) -> float:
        """Calculate weighted risk score from issues."""
        if not issues:
            return 0.0

        # Start with 0 and add weights (cap at 1.0)
        total_risk = 0.0
        for issue in issues:
            total_risk += issue.weight

        # Normalize to 0-1 range using diminishing returns
        # This prevents a few low-risk issues from inflating the score too much
        normalized_risk = 1.0 - (1.0 / (1.0 + total_risk * 0.5))

        return min(1.0, normalized_risk)

    def _get_recommended_backend(
        self,
        risk_score: float,
        issues: List[CodeIssue]
    ) -> str:
        """Determine recommended backend based on risk."""
        # Check for critical issues
        has_critical = any(issue.severity == "critical" for issue in issues)

        if has_critical:
            return "secure"

        if risk_score >= 0.5:
            return "secure"
        elif risk_score >= 0.2:
            return "fast"
        else:
            return "fast"

    def get_pattern_summary(self) -> Dict[str, Dict]:
        """Get a summary of all patterns being scanned."""
        return {
            name: {
                "weight": PATTERN_WEIGHTS[name],
                "severity": SEVERITY_MAP[name],
            }
            for name in self.DANGEROUS_PATTERNS.keys()
        }
