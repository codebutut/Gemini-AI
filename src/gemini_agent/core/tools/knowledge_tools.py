import os
import json
import networkx as nx
from typing import List, Dict, Any, Optional
from pathlib import Path
from gemini_agent.utils.logger import get_logger
from gemini_agent.core.tools import tool

logger = get_logger(__name__)

KNOWLEDGE_BASE_DIR = Path(".agent_state")
KG_PATH = KNOWLEDGE_BASE_DIR / "knowledge_graph.json"
NOTES_DIR = KNOWLEDGE_BASE_DIR / "notes"


def _ensure_dirs():
    KNOWLEDGE_BASE_DIR.mkdir(exist_ok=True)
    NOTES_DIR.mkdir(exist_ok=True)


@tool
def update_knowledge_graph(entity: str, relation: str, target: str) -> str:
    """Adds or updates a relationship in the personal knowledge graph."""
    _ensure_dirs()
    G = nx.DiGraph()
    if KG_PATH.exists():
        try:
            with open(KG_PATH, "r") as f:
                data = json.load(f)
                G = nx.node_link_graph(data)
        except Exception as e:
            logger.error(f"Error loading KG: {e}")

    G.add_edge(entity, target, relation=relation)

    try:
        data = nx.node_link_data(G)
        with open(KG_PATH, "w") as f:
            json.dump(data, f, indent=2)
        return f"Successfully added relationship: {entity} --({relation})--> {target}"
    except Exception as e:
        return f"Error saving KG: {e}"


@tool
def query_knowledge_graph(query: str) -> str:
    """Searches the knowledge graph for entities and their connections."""
    if not KG_PATH.exists():
        return "Knowledge graph is empty."

    try:
        with open(KG_PATH, "r") as f:
            data = json.load(f)
            G = nx.node_link_graph(data)

        results = []
        # Simple keyword search on nodes
        matching_nodes = [n for n in G.nodes if query.lower() in str(n).lower()]

        for node in matching_nodes:
            connections = []
            for _, target, data in G.out_edges(node, data=True):
                connections.append(
                    f"-> {data.get('relation', 'related to')} -> {target}"
                )
            for source, _, data in G.in_edges(node, data=True):
                connections.append(
                    f"<- {data.get('relation', 'related to')} <- {source}"
                )

            results.append(f"Entity: {node}\n" + "\n".join(connections))

        if not results:
            return f"No entities found matching '{query}'."

        return "\n\n".join(results)
    except Exception as e:
        return f"Error querying KG: {e}"


@tool
def create_note(title: str, content: str, tags: List[str] = None) -> str:
    """Saves a structured note to the personal knowledge base."""
    _ensure_dirs()
    filename = "".join(c for c in title if c.isalnum() or c in (" ", "_")).rstrip()
    filename = filename.replace(" ", "_").lower() + ".json"
    note_path = NOTES_DIR / filename

    note_data = {
        "title": title,
        "content": content,
        "tags": tags or [],
        "created_at": str(Path(note_path).stat().st_ctime)
        if note_path.exists()
        else None,  # Placeholder
    }

    try:
        with open(note_path, "w") as f:
            json.dump(note_data, f, indent=2)
        return f"Note '{title}' saved successfully to {note_path}."
    except Exception as e:
        return f"Error saving note: {e}"


@tool
def search_notes(query: str, tags: List[str] = None) -> str:
    """Retrieves notes based on content or tags."""
    if not NOTES_DIR.exists():
        return "No notes found."

    results = []
    for note_file in NOTES_DIR.glob("*.json"):
        try:
            with open(note_file, "r") as f:
                note = json.load(f)

            match = False
            if query and (
                query.lower() in note["title"].lower()
                or query.lower() in note["content"].lower()
            ):
                match = True
            if tags:
                if any(
                    tag.lower() in [t.lower() for t in note.get("tags", [])]
                    for tag in tags
                ):
                    match = True

            if match:
                results.append(
                    f"Title: {note['title']}\nTags: {', '.join(note.get('tags', []))}\nContent: {note['content'][:200]}..."
                )
        except Exception as e:
            logger.error(f"Error reading note {note_file}: {e}")

    if not results:
        return "No matching notes found."

    return "\n\n---\n\n".join(results)


@tool
def map_document_relationships(directory: str) -> str:
    """Analyzes files in a directory to find thematic connections (Placeholder for advanced NLP)."""
    path = Path(directory)
    if not path.exists():
        return f"Directory {directory} not found."

    files = list(path.glob("*.*"))
    if not files:
        return "No files found in directory."

    # In a real implementation, we'd use TF-IDF or embeddings.
    # For now, we'll return a list of files and suggest they are related by location.
    file_list = [f.name for f in files]
    return f"Found {len(file_list)} files in {directory}. Thematic mapping requires content analysis of: {', '.join(file_list)}"


@tool
def analyze_transcript(filepath: str) -> str:
    """Analyzes a meeting transcript to extract action items and summaries."""
    return f"Please use the ResearchAgent to read {filepath} and then ask it to summarize action items and key decisions."


@tool
def summarize_research_paper(filepath: str) -> str:
    """Provides a structured summary of a research paper."""
    return f"Please use the ResearchAgent to read {filepath} and then ask it to provide a structured summary (Abstract, Methodology, Results, Conclusion)."
