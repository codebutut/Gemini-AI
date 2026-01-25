import multiprocessing
import os
import sys
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

# Add src directory to sys.path
src_path = os.path.join(os.path.dirname(__file__), "src")
sys.path.insert(0, src_path)

from gemini_agent.main import main

if __name__ == "__main__":
    # Set multiprocessing start method as early as possible
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    try:
        main()
    except KeyboardInterrupt:
        pass
