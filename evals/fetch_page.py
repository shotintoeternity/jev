"""Download a web page as plain text, so quotes can be verified against what the page really says.

    uv run python -m evals.fetch_page <url> <out.txt>

Used when evidence is gathered inside a Claude Code session instead of through the API.
"""

import re
import sys
from html.parser import HTMLParser
from pathlib import Path

import httpx2 as httpx

SKIP = {"script", "style", "noscript", "svg", "head", "nav", "footer", "form"}
BLOCK = {"p", "div", "li", "br", "h1", "h2", "h3", "h4", "h5", "h6", "tr", "td", "th", "section", "article", "blockquote", "dd", "dt"}


class Text(HTMLParser):
    def __init__(self):
        super().__init__()
        self.out, self.skip = [], 0

    def handle_starttag(self, tag, attrs):
        if tag in SKIP:
            self.skip += 1
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_endtag(self, tag):
        if tag in SKIP and self.skip:
            self.skip -= 1
        elif tag in BLOCK:
            self.out.append("\n")

    def handle_data(self, data):
        if not self.skip:
            self.out.append(data)


def page_text(url: str) -> str:
    r = httpx.get(url, follow_redirects=True, timeout=30, headers={"User-Agent": "Mozilla/5.0 (research; jevin fact-check)"})
    r.raise_for_status()
    if "html" not in r.headers.get("content-type", "html"):
        return r.text
    p = Text()
    p.feed(r.text)
    text = "".join(p.out)
    text = re.sub(r"[ \t\r\f\v]+", " ", text)
    return re.sub(r"\n\s*\n+", "\n\n", text).strip()


if __name__ == "__main__":
    url, out = sys.argv[1], Path(sys.argv[2])
    try:
        text = page_text(url)
    except Exception as e:
        print(f"FAILED {type(e).__name__}: {e}")
        sys.exit(1)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(f"URL: {url}\n\n{text}")
    print(f"saved {len(text):,} chars to {out}")
