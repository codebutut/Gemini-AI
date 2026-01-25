# Gemini AI Agent

![Python](https://img.shields.io/badge/Python-3.10%2B-blue)
![License](https://img.shields.io/badge/License-MIT-green)
![Status](https://img.shields.io/badge/Status-Active-success)
![PyQt6](https://img.shields.io/badge/GUI-PyQt6-orange)

**Gemini AI Agent** is a professional-grade, autonomous desktop assistant powered by Google's Gemini models. Built with Python and PyQt6, it combines a modern, fluent interface with advanced capabilities like Multi-Agent Systems (MAS), Model Context Protocol (MCP), and deep local system integration.

---

## 📋 Table of Contents
- [Features](#-features)
- [Installation](#-installation)
- [Configuration](#-configuration)
- [Usage](#-usage)
- [Architecture](#-architecture)
- [Contributing](#-contributing)
- [License](#-license)

---

## 🚀 Features

### 🧠 Advanced AI Core
- **Gemini Integration**: Seamless access to Gemini Flash, Pro, and Thinking models.
- **Multi-Agent System (MAS)**: Orchestrate specialized agents (Coder, Researcher, Planner) to solve complex tasks.
- **Long-Term Memory**: ChromaDB-backed vector store for semantic context retention across sessions.

### 🖥️ Modern User Interface
- **Fluent Design**: Beautiful, responsive UI based on Windows 11 design principles (in development).
- **Project Explorer**: Integrated file browser for context management.
- **Code & Markdown**: Rich rendering of code blocks and markdown content.
- **Theme Support**: Native Dark and Light mode support.

### 🛠️ Powerful Toolset
- **System Operations**: Execute shell commands, manage processes, and monitor resources.
- **File Management**: Read/Write support for PDF, DOCX, XLSX, PPTX, and code files.
- **Web Capabilities**: Web search, URL fetching, and dynamic content analysis.
- **Code Engineering**: AST-based analysis, refactoring, and git integration.

### 🔌 Extensibility
- **MCP Support**: Connect to external Model Context Protocol servers.
- **Plugin Architecture**: Easily extend functionality with custom plugins.
- **Conductor**: Define custom automation workflows.

---

## 🛠️ Installation

### Prerequisites
- **Python 3.10** or higher.
- A **Google Gemini API Key** (Get it from [Google AI Studio](https://aistudio.google.com/)).

### Step-by-Step Guide

1. **Clone the Repository**
   ```bash
   git clone https://github.com/codebutut/Gemini-AI.git
   cd Gemini-AI
   ```

2. **Create a Virtual Environment** (Recommended)
   ```bash
   python -m venv venv
   # Windows
   venv\Scripts\activate
   # Linux/macOS
   source venv/bin/activate
   ```

3. **Install Dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Setup Environment Variables**
   Create a `.env` file in the root directory:
   ```env
   GOOGLE_API_KEY=your_actual_api_key_here
   ```

---

## ⚙️ Configuration

### `settings.json`
The application uses `settings.json` for persistent user preferences. This is automatically generated on first run but can be manually edited to configure:
- **Theme**: `Dark` or `Light`.
- **Model**: Default Gemini model version.
- **Paths**: Custom paths for downloads or workspace.

### `mcp_config.json`
Configure external MCP servers here. Example:
```json
{
  "mcpServers": {
    "filesystem": {
      "command": "npx",
      "args": ["-y", "@modelcontextprotocol/server-filesystem", "/path/to/allowed/dir"]
    }
  }
}
```

---

## 📖 Usage

### Starting the Application
```bash
python run.py
```

### Interface Guide
- **Chat Area**: Main interaction hub. Type prompts, paste code, or drag-and-drop files.
- **Sidebar**: Access History, Files, and Settings.
- **Command Palette**:
    - `/search <query>`: Search past conversations and indexed files.
    - `/mas <prompt>`: Delegate task to the Multi-Agent System.
    - `/conductor`: Open the workflow orchestrator.

---

## 🏗️ Architecture

The project is structured for modularity and scalability:

- **`src/gemini_agent/ui/`**: PyQt6 frontend components.
- **`src/gemini_agent/core/`**: Backend logic, including the Agent loop, Tool Executor, and Memory Manager.
- **`src/gemini_agent/core/mas/`**: Multi-Agent System logic.
- **`plugins/`**: Directory for user-installed extensions.

---

## 🤝 Contributing

Contributions are welcome! Please follow these steps:
1. Fork the repository.
2. Create a feature branch (`git checkout -b feature/AmazingFeature`).
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`).
4. Push to the branch (`git push origin feature/AmazingFeature`).
5. Open a Pull Request.

---

## 📄 License

Distributed under the MIT License. See `LICENSE` for more information.
