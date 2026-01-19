import ast
import difflib
import html
import io
import logging
import os
import re
import subprocess
import sys
import tempfile

logger = logging.getLogger(__name__)


class ReviewEngine:
    """
    Core logic for the Deep Review system.
    Handles diff generation, static analysis, and security scanning.
    """

    # Pre-compiled and combined security patterns for performance
    _SECURITY_PATTERN = re.compile(
        r"(?P<aws>AKIA[0-9A-Z]{16})|"
        r"(?P<api_key>api_key\s*=\s*['\"][a-zA-Z0-9_\-]{20,}['\"])|"
        r"(?P<password>password\s*=\s*['\"][^'\"]{6,}['\"])|"
        r"(?P<syscall>os\.system\(|subprocess\.call\(|eval\()",
        re.IGNORECASE,
    )

    _RISK_LABELS = {
        "aws": "AWS Key",
        "api_key": "Generic API Key",
        "password": "Hardcoded Password",
        "syscall": "Dangerous System Call",
    }

    @staticmethod
    def generate_diff_html(old_content: str, new_content: str, theme_mode: str = "Dark") -> str:
        """
        Generates a side-by-side or unified diff in HTML format using StringIO for efficiency.
        """
        if not old_content:
            old_content = ""
        if not new_content:
            new_content = ""

        old_lines = old_content.splitlines()
        new_lines = new_content.splitlines()

        diff = difflib.unified_diff(old_lines, new_lines, fromfile="Current", tofile="Proposed", lineterm="")

        # Theme-aware colors
        if theme_mode == "Dark":
            add_bg, add_fg = "#1e3a1e", "#afffbe"
            rem_bg, rem_fg = "#442a2a", "#ffa3a3"
            info_fg = "#8be9fd"
            default_fg = "#888"
        else:
            add_bg, add_fg = "#e6ffec", "#22863a"
            rem_bg, rem_fg = "#ffebe9", "#cb2431"
            info_fg = "#005cc5"
            default_fg = "#6a737d"

        # Use StringIO for efficient string building
        output = io.StringIO()
        output.write(f"<pre style='font-family: monospace; white-space: pre; color: {default_fg};'>\n")

        for line in diff:
            escaped_line = html.escape(line)
            if line.startswith("+"):
                output.write(f"<div style='color: {add_fg}; background-color: {add_bg};'>{escaped_line}</div>\n")
            elif line.startswith("-"):
                output.write(f"<div style='color: {rem_fg}; background-color: {rem_bg};'>{escaped_line}</div>\n")
            elif line.startswith("@@"):
                output.write(f"<div style='color: {info_fg}; font-weight: bold;'>{escaped_line}</div>\n")
            elif line.startswith("^"):
                output.write(f"<div style='color: {info_fg};'>{escaped_line}</div>\n")
            else:
                output.write(f"<div>{escaped_line}</div>\n")

        output.write("</pre>")
        return output.getvalue()

    @staticmethod
    def analyze_code(code: str) -> list[str]:
        """
        Performs static analysis on the code using AST and ruff (via stdin for speed).
        """
        issues: list[str] = []

        # 1. Basic AST check
        try:
            ast.parse(code)
        except SyntaxError as e:
            issues.append(f"CRITICAL: Syntax Error at line {e.lineno}: {e.msg}")
            return issues
        except Exception as e:
            issues.append(f"Error parsing code: {str(e)}")
            return issues

        # 2. Advanced analysis using Ruff (stdin)
        try:
            # Try Ruff first with stdin to avoid disk I/O
            result = subprocess.run(
                ["ruff", "check", "-", "--no-cache", "--output-format=concise"],
                input=code,
                capture_output=True,
                text=True,
                timeout=5,
            )
            if result.stdout:
                for line in result.stdout.splitlines():
                    # Ruff stdin output uses '-' as filename
                    clean_line = line.replace("-:", "code.py:").strip()
                    issues.append(f"LINT (ruff): {clean_line}")

            # If ruff succeeded (even with issues), we return
            if result.returncode in (0, 1):  # 1 means issues found
                return issues

        except FileNotFoundError:
            # Fallback to pylint if ruff is not installed (requires temp file)
            tmp_path = None
            try:
                with tempfile.NamedTemporaryFile(mode="w", suffix=".py", delete=False) as tmp:
                    tmp.write(code)
                    tmp_path = tmp.name

                result = subprocess.run(
                    [
                        sys.executable,
                        "-m",
                        "pylint",
                        "--max-line-length=120",
                        "--reports=n",
                        "--score=n",
                        "--disable=C0114,C0115,C0116",
                        tmp_path,
                    ],
                    capture_output=True,
                    text=True,
                    timeout=15,
                )

                if result.stdout:
                    for line in result.stdout.splitlines():
                        if not line.startswith("************* Module") and tmp_path in line:
                            clean_line = line.replace(tmp_path, "code.py")
                            issues.append(f"LINT (pylint): {clean_line}")
            except Exception as e:
                issues.append(f"LINT: Error running pylint: {str(e)}")
            finally:
                if tmp_path and os.path.exists(tmp_path):
                    os.remove(tmp_path)
        except subprocess.TimeoutExpired:
            issues.append("LINT: Analysis timed out.")
        except Exception as e:
            issues.append(f"LINT: Error running analysis: {str(e)}")

        return issues

    @staticmethod
    def scan_security(code: str) -> list[str]:
        """
        Scans for potential security risks using regex patterns.
        """
        risks: list[str] = []
        for match in ReviewEngine._SECURITY_PATTERN.finditer(code):
            for group_name, value in match.groupdict().items():
                if value:
                    label = ReviewEngine._RISK_LABELS.get(group_name, "Unknown Risk")
                    risks.append(f"WARNING: Potential {label} detected.")
        return list(set(risks))
