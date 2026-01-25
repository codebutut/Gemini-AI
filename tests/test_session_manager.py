import os
import pytest
from pathlib import Path
from gemini_agent.core.session_manager import SessionManager

@pytest.fixture
def session_manager(tmp_path):
    history_file = tmp_path / "history.json"
    return SessionManager(history_file)

def test_create_session(session_manager):
    session_id = session_manager.create_session(title="Test Session")
    assert session_id in session_manager.sessions
    assert session_manager.sessions[session_id].title == "Test Session"
    assert session_manager.current_session_id == session_id

def test_add_message(session_manager):
    session_id = session_manager.create_session()
    session_manager.add_message(session_id, "user", "Hello")
    session = session_manager.get_session(session_id)
    assert len(session.messages) == 1
    assert session.messages[0].text == "Hello"
    assert session.messages[0].role == "user"

def test_delete_session(session_manager):
    session_id = session_manager.create_session()
    assert session_manager.delete_session(session_id)
    assert session_id not in session_manager.sessions
    assert session_manager.current_session_id is None

def test_update_session_title(session_manager):
    session_id = session_manager.create_session(title="Old Title")
    session_manager.update_session_title(session_id, "New Title")
    assert session_manager.sessions[session_id].title == "New Title"

def test_save_and_load_history(tmp_path):
    history_file = tmp_path / "history.json"
    sm1 = SessionManager(history_file)
    sid = sm1.create_session(title="Persistent Session")
    sm1.add_message(sid, "user", "Save me")
    sm1.save_history(sync=True)
    
    sm2 = SessionManager(history_file)
    assert sid in sm2.sessions
    assert sm2.sessions[sid].title == "Persistent Session"
    assert sm2.sessions[sid].messages[0].text == "Save me"
