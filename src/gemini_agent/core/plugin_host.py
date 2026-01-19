import importlib.util
import inspect
import json
import os
import sys


def load_plugin_class(filepath: str):
    module_name = os.path.splitext(os.path.basename(filepath))[0]
    spec = importlib.util.spec_from_file_location(module_name, filepath)
    if spec and spec.loader:
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)

        # Find the Plugin class (assuming it's imported or defined in the module)
        # We need to be careful here because we don't have the base Plugin class in this process
        # unless we import it.
        for name, obj in inspect.getmembers(module):
            if inspect.isclass(obj) and name != "Plugin" and hasattr(obj, "execute_tool"):
                return obj
    return None


def main():
    if len(sys.argv) < 2:
        print("Usage: python plugin_host.py <plugin_path>")
        sys.exit(1)

    plugin_path = sys.argv[1]

    # Add src to path so we can import Plugin base class if needed
    src_path = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", ".."))
    if src_path not in sys.path:
        sys.path.insert(0, src_path)

    plugin_class = load_plugin_class(plugin_path)
    if not plugin_class:
        print(f"Error: Could not find Plugin class in {plugin_path}")
        sys.exit(1)

    plugin_instance = plugin_class()

    print("READY")  # Signal to parent process
    sys.stdout.flush()

    while True:
        line = sys.stdin.readline()
        if not line:
            break

        try:
            request = json.loads(line)
            cmd = request.get("command")

            if cmd == "get_tools":
                tools = plugin_instance.get_tools()
                # Convert types.FunctionDeclaration to dict for JSON serialization
                # This is tricky because FunctionDeclaration is a complex object
                # For now, let's assume we can serialize it or we need a helper
                serialized_tools = []
                for t in tools:
                    # Simple conversion for now, might need more robust handling
                    serialized_tools.append(
                        {
                            "name": t.name,
                            "description": t.description,
                            "parameters": t.parameters,  # This is usually a dict
                        }
                    )
                print(json.dumps({"status": "ok", "result": serialized_tools}))

            elif cmd == "execute_tool":
                tool_name = request.get("tool_name")
                args = request.get("args")
                result = plugin_instance.execute_tool(tool_name, args)
                print(json.dumps({"status": "ok", "result": result}))

            elif cmd == "exit":
                break

            sys.stdout.flush()
        except Exception as e:
            print(json.dumps({"status": "error", "message": str(e)}))
            sys.stdout.flush()


if __name__ == "__main__":
    main()
