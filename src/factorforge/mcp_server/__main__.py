"""Run the knowledge-graph MCP server as a standalone process.

    python -m factorforge.mcp_server            # stdio (the transport the Agent SDK spawns)
    python -m factorforge.mcp_server --http     # streamable-HTTP at http://host:port/mcp

The Claude Agent SDK connects to the stdio form by configuring an external MCP server:
``mcp_servers={"kg": {"command": "python", "args": ["-m", "factorforge.mcp_server"], ...}}``.
"""

from __future__ import annotations

import sys

from factorforge.config import get_settings
from factorforge.logging import configure_logging, get_logger
from factorforge.mcp_server.server import build_server

log = get_logger(__name__)


def main(argv: list[str] | None = None) -> None:
    argv = sys.argv[1:] if argv is None else argv
    settings = get_settings()
    configure_logging(settings.log_level, settings.log_json)

    server = build_server(settings=settings)
    use_http = "--http" in argv or settings.mcp_transport == "http"
    if use_http:
        log.info("mcp.serve", transport="streamable-http", host=settings.mcp_http_host, port=settings.mcp_http_port)
        server.run(transport="streamable-http")
    else:
        log.info("mcp.serve", transport="stdio")
        server.run()


if __name__ == "__main__":
    main()
