class ModeDetector:
    """
    Detects whether the user likely wants web search or local operations.
    """

    WEB_SEARCH_KEYWORDS = [
        "current",
        "latest",
        "news",
        "today",
        "yesterday",
        "recent",
        "weather",
        "stock",
        "price",
        "market",
        "sports",
        "score",
        "who is",
        "what is",
        "when was",
        "where is",
        "how to",
        "update",
        "breaking",
        "live",
        "trending",
        "covid",
        "virus",
        "president",
        "election",
        "world cup",
        "oscars",
        "nobel",
    ]

    LOCAL_OPS_KEYWORDS = [
        "file",
        "directory",
        "folder",
        "read",
        "write",
        "create",
        "delete",
        "modify",
        "edit",
        "code",
        "python",
        "script",
        "run",
        "execute",
        "debug",
        "test",
        "analyze",
        "refactor",
        "process",
        "list",
        "show",
        "display",
        "git",
        "commit",
        "install",
        "package",
        "dependency",
        "system",
        "process",
        "kill",
        "start",
        "application",
        "program",
    ]

    def detect_mode(self, prompt: str, use_grounding: bool) -> str:
        """
        Detects the mode based on the prompt and grounding setting.
        Returns: 'grounding' or 'function_calling'
        """
        prompt_lower = prompt.lower()

        web_score = sum(1 for keyword in self.WEB_SEARCH_KEYWORDS if keyword in prompt_lower)

        # If user explicitly enabled grounding, use it for web-ish queries
        if use_grounding and web_score > 0:
            return "grounding"
        # Otherwise default to function calling for everything else
        else:
            return "function_calling"
