import subprocess
from pydantic import BaseModel, Field
from . import tool, validate_args


class GitArgs(BaseModel):
    operation: str = Field(
        ..., description="git command (clone, pull, commit, push, etc.)."
    )
    args: list[str] | None = Field(None, description="Additional arguments.")


@tool
@validate_args(GitArgs)
def git_operation(operation: str, args: list[str] | None = None) -> str:
    """
    Executes Git operations.

    Args:
        operation: git command (clone, pull, commit, push, etc.).
        args: Additional arguments.

    Returns:
        str: Git operation output.
    """
    try:
        cmd = ["git", operation]
        if args:
            cmd.extend(args)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=60)
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        return f"Git {operation} completed with code {result.returncode}\n" + "\n".join(
            output
        )
    except subprocess.TimeoutExpired:
        return f"Error: Git {operation} timed out (60s limit)."
