"""
Web fetching tool for exploring URLs in user profiles
"""

import urllib.request
import urllib.error
from html.parser import HTMLParser
from io import StringIO


class HTMLTextExtractor(HTMLParser):
    """Extract text content from HTML"""

    def __init__(self):
        super().__init__()
        self.text = StringIO()
        self.skip = False

    def handle_starttag(self, tag, attrs):
        # Skip script and style tags
        if tag.lower() in ['script', 'style']:
            self.skip = True

    def handle_endtag(self, tag):
        if tag.lower() in ['script', 'style']:
            self.skip = False

    def handle_data(self, data):
        if not self.skip:
            self.text.write(data)
            self.text.write(' ')

    def get_text(self):
        return self.text.getvalue()


def fetch_url(url: str, max_length: int = 8000) -> dict:
    """
    Fetch content from a URL and extract text.

    Args:
        url: The URL to fetch
        max_length: Maximum length of returned text (default: 8000 chars)

    Returns:
        dict with keys:
            - success: bool
            - url: str (the URL fetched)
            - content: str (extracted text) if successful
            - error: str (error message) if failed
    """
    try:
        # Fetch the URL
        headers = {
            'User-Agent': 'Mozilla/5.0 (DXTR Profile Agent)'
        }
        req = urllib.request.Request(url, headers=headers)

        with urllib.request.urlopen(req, timeout=10) as response:
            # Read and decode content
            content_bytes = response.read()
            content_type = response.headers.get('Content-Type', '')

            # Try to detect encoding
            encoding = 'utf-8'
            if 'charset=' in content_type:
                encoding = content_type.split('charset=')[-1].split(';')[0].strip()

            try:
                html = content_bytes.decode(encoding)
            except UnicodeDecodeError:
                html = content_bytes.decode('utf-8', errors='ignore')

            # Extract text from HTML
            parser = HTMLTextExtractor()
            parser.feed(html)
            text = parser.get_text()

            # Clean up whitespace
            text = ' '.join(text.split())

            # Truncate if needed
            if len(text) > max_length:
                text = text[:max_length] + "... (truncated)"

            return {
                "success": True,
                "url": url,
                "content": text
            }

    except urllib.error.HTTPError as e:
        return {
            "success": False,
            "url": url,
            "error": f"HTTP Error {e.code}: {e.reason}"
        }
    except urllib.error.URLError as e:
        return {
            "success": False,
            "url": url,
            "error": f"URL Error: {e.reason}"
        }
    except Exception as e:
        return {
            "success": False,
            "url": url,
            "error": f"Error: {str(e)}"
        }


# Tool definition for Ollama function calling
TOOL_DEFINITION = {
    "type": "function",
    "function": {
        "name": "fetch_url",
        "description": "Fetch and extract text content from a web URL. Use this to explore links from the user's profile (e.g., GitHub, personal website, blog) to better understand their work and background.",
        "parameters": {
            "type": "object",
            "properties": {
                "url": {
                    "type": "string",
                    "description": "The URL to fetch (must be a valid http:// or https:// URL)"
                }
            },
            "required": ["url"]
        }
    }
}
