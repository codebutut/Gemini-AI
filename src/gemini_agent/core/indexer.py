import ast
import functools
import logging
import os
import sqlite3
from collections import defaultdict
from concurrent.futures import ProcessPoolExecutor
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger(__name__)


@dataclass
class Symbol:
    """Represents a code symbol (class, function, method) found during indexing."""

    name: str
    kind: str  # 'class', 'function', 'method'
    line: int
    file_path: str
    docstring: str | None = None
    parent: str | None = None

    def to_dict(self) -> dict[str, Any]:
        """Converts the Symbol object to a dictionary."""
        return {
            "name": self.name,
            "kind": self.kind,
            "line": self.line,
            "file_path": self.file_path,
            "docstring": self.docstring,
            "parent": self.parent,
        }


class SymbolVisitor(ast.NodeVisitor):
    """AST visitor to extract symbols from a Python file."""

    def __init__(self, file_path: str, rel_path: str) -> None:
        self.file_path = file_path
        self.rel_path = rel_path
        self.symbols: list[Symbol] = []
        self._current_class: str | None = None

    def visit_ClassDef(self, node: ast.ClassDef) -> None:
        """Visits a class definition and records it as a symbol."""
        self.symbols.append(
            Symbol(
                name=node.name,
                kind="class",
                line=node.lineno,
                file_path=self.rel_path,
                docstring=ast.get_docstring(node),
            )
        )
        old_class = self._current_class
        self._current_class = node.name
        self.generic_visit(node)
        self._current_class = old_class

    def visit_FunctionDef(self, node: ast.FunctionDef) -> None:
        """Visits a function or method definition and records it as a symbol."""
        kind = "method" if self._current_class else "function"
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                line=node.lineno,
                file_path=self.rel_path,
                docstring=ast.get_docstring(node),
                parent=self._current_class,
            )
        )
        self.generic_visit(node)

    def visit_AsyncFunctionDef(self, node: ast.AsyncFunctionDef) -> None:
        """Visits an async function or method definition and records it as a symbol."""
        kind = "method" if self._current_class else "function"
        self.symbols.append(
            Symbol(
                name=node.name,
                kind=kind,
                line=node.lineno,
                file_path=self.rel_path,
                docstring=ast.get_docstring(node),
                parent=self._current_class,
            )
        )
        self.generic_visit(node)


def _index_file_worker(file_path: str, root_dir: str) -> dict[str, Any] | None:
    """Worker function for parallel indexing. Must be top-level for pickling."""
    try:
        mtime = os.path.getmtime(file_path)

        # Check file size before reading (1MB limit for indexing)
        if os.path.getsize(file_path) > 1024 * 1024:
            return None

        with open(file_path, encoding="utf-8") as f:
            content = f.read()
            tree = ast.parse(content)

        rel_path = os.path.relpath(file_path, root_dir)
        visitor = SymbolVisitor(file_path, rel_path)
        visitor.visit(tree)

        return {
            "path": file_path,
            "mtime": mtime,
            "symbols": [s.to_dict() for s in visitor.symbols],
        }
    except Exception:
        return None


class Indexer:
    """Handles project-wide indexing of Python symbols with persistent SQLite caching."""

    def __init__(self, root_dir: str) -> None:
        self.root_dir = root_dir
        self.symbols: list[Symbol] = []
        self.name_map: dict[str, list[Symbol]] = {}
        self.trigram_index: dict[str, set[int]] = defaultdict(set)  # trigram -> set of symbol indices
        self.load_cache()

    def _get_cache_path(self) -> str:
        """Returns the path to the persistent SQLite cache file."""
        return os.path.join(self.root_dir, ".gemini_index_cache.db")

    def _init_db(self, conn: sqlite3.Connection):
        """Initializes the SQLite database schema."""
        cursor = conn.cursor()
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS files (
                path TEXT PRIMARY KEY,
                mtime REAL
            )
        """)
        cursor.execute("""
            CREATE TABLE IF NOT EXISTS symbols (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                file_path TEXT,
                name TEXT,
                kind TEXT,
                line INTEGER,
                docstring TEXT,
                parent TEXT,
                FOREIGN KEY(file_path) REFERENCES files(path) ON DELETE CASCADE
            )
        """)
        conn.commit()

    def load_cache(self) -> None:
        """Loads the index cache from SQLite."""
        cache_path = self._get_cache_path()
        if not os.path.exists(cache_path):
            return

        try:
            conn = sqlite3.connect(cache_path)
            self._init_db(conn)
            cursor = conn.cursor()

            cursor.execute("SELECT name, kind, line, file_path, docstring, parent FROM symbols")
            rows = cursor.fetchall()

            self.symbols = []
            self.name_map = {}
            for row in rows:
                s = Symbol(
                    name=row[0],
                    kind=row[1],
                    line=row[2],
                    file_path=row[3],
                    docstring=row[4],
                    parent=row[5],
                )
                self.symbols.append(s)
                self.name_map.setdefault(s.name.lower(), []).append(s)

            self._build_trigram_index()
            conn.close()
            logger.info(f"Loaded {len(self.symbols)} symbols from SQLite cache.")
        except Exception as e:
            logger.warning(f"Failed to load index cache: {e}")

    def _build_trigram_index(self) -> None:
        """Builds an in-memory trigram index for fast partial matching."""
        self.trigram_index.clear()
        for idx, s in enumerate(self.symbols):
            name = s.name.lower()
            if len(name) < 3:
                self.trigram_index[name].add(idx)
                continue
            for i in range(len(name) - 2):
                trigram = name[i : i + 3]
                self.trigram_index[trigram].add(idx)

    def index_project(self) -> None:
        """Recursively scans the project directory for Python files and indexes symbols in parallel."""
        cache_path = self._get_cache_path()
        conn = sqlite3.connect(cache_path)
        self._init_db(conn)
        cursor = conn.cursor()

        # Get existing file mtimes from DB
        cursor.execute("SELECT path, mtime FROM files")
        db_files = {row[0]: row[1] for row in cursor.fetchall()}

        files_to_index = []
        current_files = set()

        for root, dirs, files in os.walk(self.root_dir):
            dirs[:] = [
                d for d in dirs if not d.startswith(".") and d not in ("env", "venv", "__pycache__", "node_modules")
            ]
            for file in files:
                if file.endswith(".py"):
                    file_path = os.path.abspath(os.path.join(root, file))
                    current_files.add(file_path)
                    try:
                        mtime = os.path.getmtime(file_path)
                        if file_path not in db_files or db_files[file_path] != mtime:
                            files_to_index.append(file_path)
                    except OSError:
                        continue

        # Parallel indexing for new/changed files
        if files_to_index:
            logger.info(f"Indexing {len(files_to_index)} files in parallel...")
            worker_func = functools.partial(_index_file_worker, root_dir=self.root_dir)
            with ProcessPoolExecutor() as executor:
                results = list(executor.map(worker_func, files_to_index))

            for res in results:
                if res:
                    path = res["path"]
                    # Delete old symbols for this file
                    cursor.execute("DELETE FROM symbols WHERE file_path = ?", (path,))
                    # Insert new file info
                    cursor.execute(
                        "INSERT OR REPLACE INTO files (path, mtime) VALUES (?, ?)",
                        (path, res["mtime"]),
                    )
                    # Insert new symbols
                    for s_dict in res["symbols"]:
                        cursor.execute(
                            """
                            INSERT INTO symbols (file_path, name, kind, line, docstring, parent)
                            VALUES (?, ?, ?, ?, ?, ?)
                        """,
                            (
                                path,
                                s_dict["name"],
                                s_dict["kind"],
                                s_dict["line"],
                                s_dict["docstring"],
                                s_dict["parent"],
                            ),
                        )
            conn.commit()

        # Clean up deleted files
        deleted_files = set(db_files.keys()) - current_files
        if deleted_files:
            for path in deleted_files:
                cursor.execute("DELETE FROM files WHERE path = ?", (path,))
                cursor.execute("DELETE FROM symbols WHERE file_path = ?", (path,))
            conn.commit()

        if files_to_index or deleted_files:
            # Reload everything into memory if changes occurred
            cursor.execute("SELECT name, kind, line, file_path, docstring, parent FROM symbols")
            rows = cursor.fetchall()
            self.symbols = []
            self.name_map = {}
            for row in rows:
                s = Symbol(
                    name=row[0],
                    kind=row[1],
                    line=row[2],
                    file_path=row[3],
                    docstring=row[4],
                    parent=row[5],
                )
                self.symbols.append(s)
                self.name_map.setdefault(s.name.lower(), []).append(s)
            self._build_trigram_index()

        conn.close()

    def search(self, query: str) -> list[Symbol]:
        """Searches for symbols matching the query string using trigram index."""
        query = query.lower()
        if not query:
            return []

        # Exact match O(1)
        if query in self.name_map:
            return self.name_map[query]

        if len(query) < 3:
            # Fallback for very short queries
            return [s for s in self.symbols if query in s.name.lower()]

        # Trigram search
        potential_indices = None
        for i in range(len(query) - 2):
            trigram = query[i : i + 3]
            matches = self.trigram_index.get(trigram, set())
            if potential_indices is None:
                potential_indices = matches.copy()
            else:
                potential_indices &= matches
            if not potential_indices:
                break

        if not potential_indices:
            return []

        # Final verification (filter out false positives from trigram intersection)
        return [self.symbols[idx] for idx in potential_indices if query in self.symbols[idx].name.lower()]

    def get_all_symbols(self) -> list[Symbol]:
        """Returns all indexed symbols."""
        return self.symbols
