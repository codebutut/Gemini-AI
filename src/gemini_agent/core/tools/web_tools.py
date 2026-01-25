import json
import asyncio
from typing import Any
from pydantic import BaseModel, Field
from . import tool, validate_args


class FetchUrlArgs(BaseModel):
    url: str = Field(..., description="URL to fetch.")
    method: str = Field("GET", description="HTTP method.")
    data: dict[str, Any] | None = Field(None, description="POST data (if any).")


class FetchDynamicUrlArgs(BaseModel):
    url: str = Field(..., description="URL to fetch.")
    wait_for_selector: str | None = Field(None, description="CSS selector to wait for.")
    timeout: int = Field(30000, description="Timeout in milliseconds.")


@tool
@validate_args(FetchUrlArgs)
def fetch_url(url: str, method: str = "GET", data: dict[str, Any] | None = None) -> str:
    """
    Fetches data from a URL using standard HTTP request.

    Args:
        url: URL to fetch.
        method: HTTP method.
        data: POST data (if any).

    Returns:
        str: Response content or error message.
    """
    try:
        import httpx

        headers = {"User-Agent": "Mozilla/5.0"}
        if method.upper() == "GET":
            response = httpx.get(url, headers=headers, timeout=10)
        elif method.upper() == "POST":
            response = httpx.post(url, json=data, headers=headers, timeout=10)
        else:
            return f"Unsupported HTTP method: {method}"
        response.raise_for_status()
        try:
            return json.dumps(response.json(), indent=2)
        except:
            return response.text[:10000]
    except Exception as e:
        return f"Request Error: {str(e)}"


@tool
@validate_args(FetchDynamicUrlArgs)
def fetch_dynamic_url(
    url: str, wait_for_selector: str | None = None, timeout: int = 30000
) -> str:
    """
    Fetches data from a URL that requires JavaScript rendering (dynamic content).

    Args:
        url: URL to fetch.
        wait_for_selector: CSS selector to wait for before returning.
        timeout: Timeout in milliseconds.

    Returns:
        str: Rendered page content or error message.
    """
    try:
        from playwright.sync_api import sync_playwright

        with sync_playwright() as p:
            browser = p.chromium.launch(headless=True)
            page = browser.new_page()
            page.goto(url, timeout=timeout)

            if wait_for_selector:
                page.wait_for_selector(wait_for_selector, timeout=timeout)

            content = page.content()
            # Extract text content for better readability if it's a large page
            text_content = page.evaluate("() => document.body.innerText")
            browser.close()

            return text_content[:15000] if text_content else content[:15000]
    except ImportError:
        return (
            "Error: 'playwright' is not installed. Please install it to use this tool."
        )
    except Exception as e:
        return f"Dynamic Fetch Error: {str(e)}"
