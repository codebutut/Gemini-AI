import inspect
import re
from collections.abc import Callable
from typing import Any, Union

from google.genai import types


def get_type_map(python_type: Any) -> types.Type:
    """Maps Python types to Gemini API types."""
    mapping = {
        str: types.Type.STRING,
        int: types.Type.INTEGER,
        float: types.Type.NUMBER,
        bool: types.Type.BOOLEAN,
        list: types.Type.ARRAY,
        dict: types.Type.OBJECT,
    }
    # Handle generic types and Union/Optional
    origin = getattr(python_type, "__origin__", None)
    if origin in (list, list):
        return types.Type.ARRAY
    if origin in (dict, dict):
        return types.Type.OBJECT
    if origin is Union:
        args = getattr(python_type, "__args__", [])
        # If it's Optional[T], args will be (T, type(None))
        non_none_args = [a for a in args if a is not type(None)]
        if non_none_args:
            return get_type_map(non_none_args[0])

    return mapping.get(python_type, types.Type.STRING)


def auto_generate_declaration(func: Callable) -> types.FunctionDeclaration:
    """Automatically generates a FunctionDeclaration from a Python function."""
    sig = inspect.signature(func)
    doc = inspect.getdoc(func) or "No description provided."

    # Parse parameter descriptions from docstring (simple version)
    param_docs = {}
    doc_lines = doc.split("\n")
    current_param = None
    for line in doc_lines:
        line = line.strip()
        if line.startswith("Args:"):
            continue
        match = re.match(r"^(\w+):\s*(.*)", line)
        if match:
            current_param = match.group(1)
            param_docs[current_param] = match.group(2)
        elif current_param and line:
            param_docs[current_param] += " " + line

    properties = {}
    required = []

    def unwrap(t):
        orig = getattr(t, "__origin__", None)
        if orig is Union:
            args = getattr(t, "__args__", [])
            non_none = [a for a in args if a is not type(None)]
            return unwrap(non_none[0]) if non_none else t
        return t

    for name, param in sig.parameters.items():
        # Skip self/cls if any (though not expected here)
        if name in ("self", "cls"):
            continue

        param_type = get_type_map(param.annotation)

        schema_args = {"type": param_type}
        schema_args["description"] = param_docs.get(name, f"Parameter {name}")

        # Handle arrays
        if param_type == types.Type.ARRAY:
            # Try to guess item type from annotation
            item_type = types.Type.STRING

            unwrapped = unwrap(param.annotation)
            orig = getattr(unwrapped, "__origin__", None)
            if orig in (list, list):
                inner_args = getattr(unwrapped, "__args__", [])
                if inner_args:
                    item_type = get_type_map(inner_args[0])

            schema_args["items"] = types.Schema(type=item_type)

        properties[name] = types.Schema(**schema_args)

        if param.default is inspect.Parameter.empty:
            required.append(name)

    return types.FunctionDeclaration(
        name=func.__name__,
        description=doc_lines[0] if doc_lines else "No description",
        parameters=types.Schema(
            type=types.Type.OBJECT, properties=properties, required=required
        ),
    )
