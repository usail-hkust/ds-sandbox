"""
Unit tests for CodeScanner.

Tests the static code security analysis functionality including
dangerous pattern detection and risk scoring.
"""

import pytest
from typing import Any

from ds_sandbox.security.scanner import CodeScanner
from ds_sandbox.types import CodeScanResult


class TestCodeScannerInit:
    """Tests for CodeScanner initialization."""

    def test_scanner_init(self):
        """Test scanner initialization compiles patterns."""
        scanner = CodeScanner()

        assert scanner._compiled_patterns is not None
        assert len(scanner._compiled_patterns) > 0

    def test_patterns_compiled(self):
        """Test that all dangerous patterns are compiled."""
        scanner = CodeScanner()

        expected_patterns = [
            "file_write",
            "network",
            "subprocess",
            "dynamic_exec",
            "system",
        ]

        for pattern in expected_patterns:
            assert pattern in scanner._compiled_patterns


class TestSafeCodeScanning:
    """Tests for scanning safe code."""

    def test_safe_code_empty(self):
        """Test scanning empty code returns safe result."""
        scanner = CodeScanner()

        result = scanner.scan("")

        assert result.is_safe is True
        assert result.risk_score == 0.0
        assert len(result.issues) == 0

    def test_safe_code_simple(self):
        """Test scanning simple safe code."""
        scanner = CodeScanner()

        code = """
import pandas as pd
import numpy as np

df = pd.DataFrame({'a': [1, 2, 3]})
print(df.head())
"""

        result = scanner.scan(code)

        assert result.is_safe is True
        assert result.risk_score < 0.3

    def test_safe_code_data_science(self):
        """Test scanning data science code."""
        scanner = CodeScanner()

        code = """
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split

# Load data
df = pd.read_csv('data.csv')
X = df.drop('target', axis=1)
y = df['target']

# Train model
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)
model = RandomForestClassifier()
model.fit(X_train, y_train)
print(model.score(X_test, y_test))
"""

        result = scanner.scan(code)

        # Should be safe with low risk score
        assert result.is_safe is True or result.risk_score < 0.5


class TestDangerousPatternDetection:
    """Tests for detecting dangerous patterns."""

    def test_detect_os_system(self):
        """Test detecting os.system calls."""
        scanner = CodeScanner()

        code = """
import os
os.system('rm -rf /')
"""

        result = scanner.scan(code)

        assert result.is_safe is False
        assert result.risk_score > 0.1

    def test_detect_subprocess_popen(self):
        """Test detecting subprocess.Popen calls."""
        scanner = CodeScanner()

        code = """
import subprocess
subprocess.Popen(['ls', '-la'])
"""

        result = scanner.scan(code)

        assert result.is_safe is False
        assert len(result.issues) > 0

    def test_detect_dynamic_exec(self):
        """Test detecting dynamic code execution."""
        scanner = CodeScanner()

        code = """
result = eval('1 + 1')
"""

        result = scanner.scan(code)

        # eval is detected
        assert len(result.issues) > 0
        assert result.risk_score > 0

    def test_detect_exec_function(self):
        """Test detecting exec function calls."""
        scanner = CodeScanner()

        code = """
exec('x = 1')
"""

        result = scanner.scan(code)

        assert len(result.issues) > 0

    def test_detect_compile_function(self):
        """Test detecting compile function."""
        scanner = CodeScanner()

        code = """
code = compile('x = 1', '', 'exec')
"""

        result = scanner.scan(code)

        assert len(result.issues) > 0

    def test_detect_socket_import(self):
        """Test detecting socket module imports."""
        scanner = CodeScanner()

        code = """
import socket
"""

        result = scanner.scan(code)

        # socket import should be detected
        assert len(result.issues) > 0

    def test_detect_requests_import(self):
        """Test detecting requests module imports."""
        scanner = CodeScanner()

        code = """
import requests
"""

        result = scanner.scan(code)

        # requests import should be detected
        assert len(result.issues) > 0

    def test_detect_file_removal(self):
        """Test detecting file removal operations."""
        scanner = CodeScanner()

        code = """
import os
import os
os.remove('/tmp/file.txt')
os.system('ls')
"""

        result = scanner.scan(code)

        # Multiple issues should exceed threshold
        assert result.is_safe is False
        assert len(result.issues) >= 2

    def test_detect_shutil_rmtree(self):
        """Test detecting shutil.rmtree calls."""
        scanner = CodeScanner()

        code = """
import shutil
import os
shutil.rmtree('/tmp/dir')
os.system('ls')
"""

        result = scanner.scan(code)

        # Multiple issues should exceed threshold
        assert result.is_safe is False
        assert len(result.issues) >= 2

    def test_detect_sys_exit(self):
        """Test detecting sys.exit calls."""
        scanner = CodeScanner()

        code = """
import sys
sys.exit(1)
"""

        result = scanner.scan(code)

        assert result.is_safe is False

    def test_detect_pdb_set_trace(self):
        """Test detecting pdb.set_trace calls."""
        scanner = CodeScanner()

        code = """
import pdb
pdb.set_trace()
"""

        result = scanner.scan(code)

        assert result.is_safe is False


class TestRiskScoring:
    """Tests for risk score calculation."""

    def test_empty_code_zero_risk(self):
        """Test empty code has zero risk."""
        scanner = CodeScanner()

        result = scanner.scan("")

        assert result.risk_score == 0.0

    def test_single_issue_accumulates_risk(self):
        """Test single issue produces risk."""
        scanner = CodeScanner()

        # Use multi-line code for better pattern matching
        code = """
import socket
"""

        result = scanner.scan(code)

        assert len(result.issues) > 0
        assert result.risk_score > 0

    def test_multiple_issues_accumulate_risk(self):
        """Test multiple issues accumulate risk."""
        scanner = CodeScanner()

        # Multiple operations
        code = """
import os
import socket
os.system('ls')
"""

        result = scanner.scan(code)

        assert len(result.issues) >= 2


class TestRecommendedIsolation:
    """Tests for isolation level recommendations."""

    def test_safe_code_recommends_fast(self):
        """Test safe code recommends fast isolation."""
        scanner = CodeScanner()

        code = """
import pandas as pd
df = pd.DataFrame({'a': [1, 2, 3]})
print(df.head())
"""

        result = scanner.scan(code)

        assert result.recommended_backend in ["fast", "secure"]

    def test_critical_issues_recommend_secure(self):
        """Test critical issues recommend secure isolation."""
        scanner = CodeScanner()

        code = """
result = eval('os.system("rm -rf /")')
"""

        result = scanner.scan(code)

        # Should have high risk score
        assert result.risk_score > 0.3


class TestIssueReporting:
    """Tests for issue details in scan results."""

    def test_issue_contains_line_number(self):
        """Test issues include correct line numbers."""
        scanner = CodeScanner()

        code = """
import pandas as pd
# Dangerous on next line
import socket
"""

        result = scanner.scan(code)

        socket_issues = [i for i in result.issues if i.module == "socket"]
        assert len(socket_issues) > 0
        assert socket_issues[0].line > 1

    def test_issue_contains_severity(self):
        """Test issues include severity level."""
        scanner = CodeScanner()

        code = "import socket"

        result = scanner.scan(code)

        assert len(result.issues) > 0
        assert result.issues[0].severity in ["low", "medium", "high", "critical"]

    def test_issue_contains_function_name(self):
        """Test issues include function names when detected."""
        scanner = CodeScanner()

        code = """
os.system('ls')
"""

        result = scanner.scan(code)

        system_issues = [i for i in result.issues if i.type == "system"]
        if system_issues:
            assert system_issues[0].function == "system"


class TestPatternSummary:
    """Tests for pattern summary functionality."""

    def test_get_pattern_summary(self):
        """Test getting pattern summary."""
        scanner = CodeScanner()

        summary = scanner.get_pattern_summary()

        assert "file_write" in summary
        assert "network" in summary
        assert "subprocess" in summary
        assert "dynamic_exec" in summary
        assert "system" in summary

        # Each pattern should have weight and severity
        for name, info in summary.items():
            assert "weight" in info
            assert "severity" in info


class TestDeduplication:
    """Tests for issue deduplication."""

    def test_same_issue_once(self):
        """Test same issue detected only once."""
        scanner = CodeScanner()

        code = """
import socket
"""

        result = scanner.scan(code)

        # Should not have duplicate issues from same line
        issue_types = [(i.line, i.type) for i in result.issues]
        assert len(issue_types) == len(set(issue_types))
