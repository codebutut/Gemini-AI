import argparse
import asyncio
import logging
import multiprocessing
import sys
import os

# Add src directory to sys.path to allow importing gemini_agent
src_path = os.path.join(os.path.dirname(__file__), "src")
if src_path not in sys.path:
    sys.path.insert(0, src_path)

from qasync import QEventLoop
from PyQt6.QtWidgets import QApplication

from gemini_agent.config.app_config import AppConfig, setup_logging
from gemini_agent.core.attachment_manager import AttachmentManager
from gemini_agent.core.checkpoint_manager import CheckpointManager
from gemini_agent.core.conductor_manager import ConductorManager
from gemini_agent.core.extension_manager import ExtensionManager
from gemini_agent.core.indexer import Indexer
from gemini_agent.core.recent_manager import RecentManager
from gemini_agent.core.session_manager import SessionManager
from gemini_agent.core.vector_store import VectorStore
from gemini_agent.ui.theme_manager import ThemeManager
from gemini_agent.ui.fluent_main import ModernGeminiBrowser

def main():
    setup_logging()
    
    if sys.platform != "win32":
        try:
            multiprocessing.set_start_method("spawn", force=True)
        except RuntimeError:
            pass

    app = QApplication(sys.argv)
    loop = QEventLoop(app)
    asyncio.set_event_loop(loop)

    config = AppConfig()
    theme_mgr = ThemeManager(app)
    session_mgr = SessionManager(AppConfig.HISTORY_FILE)
    attachment_mgr = AttachmentManager()
    conductor_mgr = ConductorManager(extension_path=config.conductor_path)
    indexer = Indexer(root_dir=".")
    checkpoint_mgr = CheckpointManager()
    vector_store = VectorStore()
    recent_mgr = RecentManager()
    extension_mgr = ExtensionManager()
    extension_mgr.discover_plugins()

    window = ModernGeminiBrowser(
        config,
        theme_mgr,
        session_mgr,
        attachment_mgr,
        conductor_mgr,
        indexer,
        extension_mgr,
        checkpoint_mgr,
        vector_store,
        recent_mgr,
    )
    window.show()

    with loop:
        loop.run_forever()

if __name__ == "__main__":
    main()
