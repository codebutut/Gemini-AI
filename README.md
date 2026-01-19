# Gemini AI Agent 

A powerful, extensible, and professional-grade AI Agent application built with Python 3.10+ and PyQt6. This application integrates Google's Gemini AI models with a comprehensive suite of tools, a Multi-Agent System (MAS), and Model Context Protocol (MCP) support to provide a robust environment for automation, coding assistance, and data analysis.

## 🚀 Features

### 🖥️ Modern GUI
- **Intuitive Chat Interface**: Clean and responsive chat UI with Markdown support and syntax highlighting.
- **Sidebar Management**: Easily manage multiple chat sessions, view recent files, and explore project structures.
- **Project Explorer**: Integrated file browser to quickly attach files or folders to the AI context.
- **Symbol Browser**: Deep indexing of Python projects to find and reference classes, functions, and variables.
- **Terminal Integration**: Built-in terminal to monitor tool execution and system output.
- **Theme Support**: Fully customizable Dark and Light modes.

### 🧠 Advanced AI Capabilities
- **Gemini Integration**: Powered by the latest Google Gemini models (Flash, Pro, Thinking).
- **Multi-Agent System (MAS)**: Orchestrate specialized sub-agents (Research, Code, File, etc.) for complex, multi-step tasks.
- **Long-term Memory**: Semantic search powered by ChromaDB vector store for retrieving relevant information from past interactions.
- **Context Management**: Automatic handling of large contexts, including file attachments and session history.

### 🛠️ Comprehensive Toolset
The agent comes equipped with a wide array of built-in tools:
- **File Operations**: Read, write, search, and manage files across various formats (PDF, DOCX, XLSX, PPTX, etc.).
- **System Control**: Monitor processes, execute shell commands, and manage system resources.
- **Code Analysis**: Analyze Python code for complexity, style, and potential issues; perform automated refactoring.
- **Git Integration**: Full support for Git operations (clone, commit, push, pull, branch management).
- **Web & Data**: Fetch URLs, perform web searches, and execute SQL queries.
- **Visualization**: Generate charts and plots (Line, Bar, Scatter, Pie) from data.
- **Knowledge Management**: Maintain a personal knowledge graph and structured notes.

### 🔌 Extensibility
- **Model Context Protocol (MCP)**: Seamlessly connect to external MCP servers to expand the agent's capabilities.
- **Plugin System**: Install and manage third-party plugins to add new features and tools.
- **Conductor Orchestrator**: Define and execute custom workflows and domain-specific commands.

## 🛠️ Installation

### Prerequisites
- Python 3.10 or higher
- A Google Gemini API Key

### Setup
1. **Clone the repository**:
   ```bash
   git clone https://github.com/codebutut/Gemini-AI.git
   cd Gemini-AI
   ```

2. **Install dependencies**:
   ```bash
   pip install -r requirements.txt
   ```

3. **Configure Environment**:
   Create a `.env` file in the root directory and add your Gemini API key:
   ```env
   GOOGLE_API_KEY=your_api_key_here
   ```

4. **Run the application**:
   ```bash
   python run.py
   ```

## 📖 Usage Guide

### Basic Interaction
- Type your prompt in the input field at the bottom and press Enter or click the send icon.
- Use the `+` button to attach files or folders to the conversation.

### Special Commands
- `/search <query>`: Perform a semantic search across your chat history and indexed documents.
- `/mas`: Explicitly trigger the Multi-Agent System for the current prompt.
- `/conductor`: Open the Conductor Orchestrator to run predefined workflows.
- `/clear`: Clear the current chat display.

### Managing Extensions
You can manage extensions via the GUI (Settings -> Manage Plugins) or the CLI:
```bash
python run.py extension list
python run.py extension install-plugin <package-name>
python run.py extension add-mcp <name> <command> --args <args>
```

## 🏗️ Architecture

The project follows a modular architecture:
- **`src/gemini_agent/ui/`**: Contains all PyQt6 components and theme management.
- **`src/gemini_agent/core/`**: The heart of the application, including:
    - `worker.py`: Asynchronous AI interaction logic.
    - `tool_executor.py`: Safe execution of registered tools.
    - `mas/`: Orchestration logic for multiple agents.
    - `extension_manager.py`: Handling of plugins and MCP servers.
- **`conductor/`**: Configuration and scripts for automated workflows.


