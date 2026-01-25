import sys
import subprocess
from pydantic import BaseModel, Field
from . import tool, validate_args


class InstallPackageArgs(BaseModel):
    package_name: str = Field(..., description="Name of package to install.")
    upgrade: bool = Field(False, description="Whether to upgrade if already installed.")
    dev: bool = Field(False, description="Whether to install dev dependencies.")


@tool
@validate_args(InstallPackageArgs)
def install_package(package_name: str, upgrade: bool = False, dev: bool = False) -> str:
    """
    Installs Python packages using pip.

    Args:
        package_name: Name of package to install.
        upgrade: Whether to upgrade if already installed.
        dev: Whether to install dev dependencies.

    Returns:
        str: Installation output.
    """
    try:
        cmd = [sys.executable, "-m", "pip", "install"]
        if upgrade:
            cmd.append("--upgrade")
        cmd.append(package_name + "[dev]" if dev else package_name)
        result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        return "Package installation completed\n" + "\n".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Package installation timed out (300s limit)."
