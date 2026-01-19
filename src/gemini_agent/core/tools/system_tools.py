import os
import subprocess
import sys
import platform
from pathlib import Path
from typing import Any
import psutil
import pyperclip
from pydantic import BaseModel, Field
from . import tool, validate_args

class CodeArgs(BaseModel):
    code: str = Field(..., description="The Python code snippet to execute.")

class StartAppArgs(BaseModel):
    app_path: str = Field(..., description="Path to the application executable.")
    args: list[str] | None = Field(None, description="Command line arguments.")
    wait: bool = Field(False, description="Whether to wait for the application to complete.")

class KillProcessArgs(BaseModel):
    process_name: str = Field(..., description="Name or PID of process to kill.")
    force: bool = Field(False, description="Force kill if process doesn't respond.")

class CommandArgs(BaseModel):
    command: str = Field(..., description="The shell command to execute.")

class ClipboardArgs(BaseModel):
    text: str = Field(..., description="The text to set to the clipboard.")

class ProcessDetailsArgs(BaseModel):
    pid: int = Field(..., description="The PID of the process.")

@tool
@validate_args(CodeArgs)
def run_python(code: str) -> str:
    """
    Executes Python code in a separate process and returns stdout/stderr.
    Useful for calculations, data processing, or running generated scripts.

    Args:
        code: The Python code snippet to execute.

    Returns:
        str: Combined stdout and stderr or an error message.
    """
    try:
        # Performance: Use stdin instead of temporary file to reduce I/O overhead
        result = subprocess.run([sys.executable, "-"], input=code, capture_output=True, text=True, timeout=30)

        output = ""
        if result.stdout:
            output += f"Output:\n{result.stdout}\n"
        if result.stderr:
            output += f"Errors:\n{result.stderr}\n"

        return output if output else "(No output to stdout/stderr)"
    except subprocess.TimeoutExpired:
        return "Error: Code execution timed out (30s limit)."
    except Exception as e:
        return f"Error: {str(e)}"

@tool
@validate_args(StartAppArgs)
def start_application(app_path: str, args: list[str] | None = None, wait: bool = False) -> str:
    """
    Starts a local application.

    Args:
        app_path: Path to the application executable.
        args: Command line arguments.
        wait: Whether to wait for the application to complete.

    Returns:
        str: Status message or execution output.
    """
    try:
        path = Path(app_path)
        if not path.exists():
            return f"Error: Application '{app_path}' not found."

        cmd = [str(path)]
        if args:
            cmd.extend(args)

        if wait:
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            output = []
            if result.stdout:
                output.append(f"STDOUT:\n{result.stdout}")
            if result.stderr:
                output.append(f"STDERR:\n{result.stderr}")
            return f"Application completed with code {result.returncode}\n" + "\n".join(output)
        else:
            if platform.system() == "Windows":
                subprocess.Popen(cmd, creationflags=subprocess.CREATE_NEW_PROCESS_GROUP)
            else:
                subprocess.Popen(cmd, start_new_session=True)
            return f"Application started in background: {app_path}"
    except subprocess.TimeoutExpired:
        return "Error: Application execution timed out (300s limit)."

@tool
@validate_args(KillProcessArgs)
def kill_process(process_name: str, force: bool = False) -> str:
    """
    Kills a running process by name or PID.

    Args:
        process_name: Name or PID of process to kill.
        force: Force kill if process doesn't respond.

    Returns:
        str: Success or error message.
    """
    try:
        try:
            pid = int(process_name)
            proc = psutil.Process(pid)
            if force:
                proc.kill()
            else:
                proc.terminate()
            return f"Terminated process {pid}"
        except (ValueError, psutil.NoSuchProcess):
            killed = []
            for proc in psutil.process_iter(["pid", "name"]):
                try:
                    if process_name.lower() in proc.info["name"].lower():
                        if force:
                            proc.kill()
                        else:
                            proc.terminate()
                        killed.append(str(proc.info["pid"]))
                except (psutil.NoSuchProcess, psutil.AccessDenied):
                    continue

            return (
                f"Terminated processes: {', '.join(killed)}"
                if killed
                else f"No processes found with name or PID: {process_name}"
            )
    except Exception as e:
        return f"Error killing process: {str(e)}"

@tool
def list_processes() -> str:
    """
    Lists all running processes with resource usage.

    Returns:
        str: Formatted list of top 50 processes.
    """
    try:
        processes = []
        for proc in psutil.process_iter(["pid", "name", "cpu_percent", "memory_percent"]):
            try:
                info = proc.info
                processes.append(
                    f"PID: {info['pid']} | {info['name']} | CPU: {info['cpu_percent']:.1f}% | MEM: {info['memory_percent']:.1f}%"
                )
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                continue
        return "\n".join(processes[:50])
    except Exception as e:
        return f"Error listing processes: {str(e)}"

@tool
@validate_args(CommandArgs)
def execute_command(command: str) -> str:
    """
    Executes a shell command and returns the output.
    Use with caution.

    Args:
        command: The shell command to execute.

    Returns:
        str: Command output or error message.
    """
    try:
        result = subprocess.run(command, shell=True, capture_output=True, text=True, timeout=60)
        output = []
        if result.stdout:
            output.append(f"STDOUT:\n{result.stdout}")
        if result.stderr:
            output.append(f"STDERR:\n{result.stderr}")
        return f"Command exited with code {result.returncode}\n" + "\n".join(output)
    except subprocess.TimeoutExpired:
        return "Error: Command execution timed out (60s limit)."
    except Exception as e:
        return f"Error executing command: {str(e)}"

@tool
def get_clipboard() -> str:
    """
    Retrieves the current text from the system clipboard.

    Returns:
        str: Clipboard text content.
    """
    try:
        return pyperclip.paste()
    except Exception as e:
        return f"Error reading clipboard: {str(e)}"

@tool
@validate_args(ClipboardArgs)
def set_clipboard(text: str) -> str:
    """
    Sets the system clipboard to the specified text.

    Args:
        text: The text to set to the clipboard.

    Returns:
        str: Success or error message.
    """
    try:
        pyperclip.copy(text)
        return "Successfully copied text to clipboard."
    except Exception as e:
        return f"Error setting clipboard: {str(e)}"

@tool
@validate_args(ProcessDetailsArgs)
def get_process_details(pid: int) -> str:
    """
    Returns detailed information about a specific process.

    Args:
        pid: The PID of the process.

    Returns:
        str: Detailed process information.
    """
    try:
        proc = psutil.Process(pid)
        with proc.oneshot():
            info = proc.as_dict(attrs=['pid', 'name', 'status', 'cpu_percent', 'memory_info', 'create_time', 'cmdline', 'username'])
            
        details = [
            f"PID: {info['pid']}",
            f"Name: {info['name']}",
            f"Status: {info['status']}",
            f"CPU Usage: {info['cpu_percent']}%",
            f"Memory: {info['memory_info'].rss / 1024 / 1024:.2f} MB (RSS)",
            f"User: {info['username']}",
            f"Command: {' '.join(info['cmdline']) if info['cmdline'] else 'N/A'}"
        ]
        
        try:
            connections = proc.connections()
            if connections:
                details.append(f"Connections: {len(connections)}")
        except:
            pass
            
        return "\n".join(details)
    except psutil.NoSuchProcess:
        return f"Error: Process with PID {pid} not found."
    except Exception as e:
        return f"Error getting process details: {str(e)}"
