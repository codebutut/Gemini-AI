import pytest
from unittest.mock import MagicMock, patch
from gemini_agent.core.controller import ChatController
from gemini_agent.config.app_config import AppConfig

@pytest.fixture
def mock_services():
    return {
        "app_config": MagicMock(spec=AppConfig),
        "session_manager": MagicMock(),
        "attachment_manager": MagicMock(),
        "conductor_manager": MagicMock(),
        "indexer": MagicMock(),
        "extension_manager": MagicMock(),
        "checkpoint_manager": MagicMock(),
        "vector_store": MagicMock(),
    }

@pytest.fixture
def controller(mock_services):
    return ChatController(**mock_services)

def test_controller_initialization(controller):
    assert controller.worker is None
    assert controller.worker_thread is None

def test_stop_worker(controller):
    mock_thread = MagicMock()
    mock_thread.isRunning.return_value = True
    controller.worker_thread = mock_thread
    
    controller.stop_worker()
    
    mock_thread.stop.assert_called_once()
    mock_thread.wait.assert_called_once()
    assert controller.worker is None
    assert controller.worker_thread is None

@patch("gemini_agent.core.controller.GeminiWorker")
@patch("gemini_agent.core.controller.GeminiWorkerThread")
def test_send_message_no_api_key(mock_thread, mock_worker, controller):
    controller.app_config.api_key = None
    
    # Use a list to capture signal emissions
    emitted_errors = []
    controller.error_occurred.connect(emitted_errors.append)
    
    controller.send_message("Hello")
    
    assert "Enter API Key in Settings." in emitted_errors
    mock_worker.assert_not_called()

@patch("gemini_agent.core.controller.GeminiWorker")
@patch("gemini_agent.core.controller.GeminiWorkerThread")
def test_send_message_success(mock_thread, mock_worker, controller):
    controller.app_config.api_key = "fake_key"
    controller.session_manager.current_session_id = "sess_1"
    mock_session = MagicMock()
    mock_session.messages = []
    mock_session.config = {}
    controller.session_manager.get_session.return_value = mock_session
    
    controller.send_message("Hello")
    
    controller.session_manager.update_session_title.assert_called()
    controller.session_manager.add_message.assert_called()
    mock_worker.assert_called_once()
    mock_thread.assert_called_once()
