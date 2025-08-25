# Claude MCP Web Search Agent

A Model Context Protocol (MCP) integration that provides Claude with real-time web search capabilities using the Brave Search API.

## Features

- **Real-time Web Search**: Integrates with Brave Search API for current information
- **MCP Integration**: Seamless Claude tool calling via `fetch_web_content`
- **Query Enhancement**: Automatic typo correction and synonym expansion
- **Smart Reranking**: TF-IDF style scoring for better result relevance
- **Caching**: SQLite-based cache with configurable TTL
- **Retry Logic**: Exponential backoff with jitter for reliability
- **Structured Logging**: Comprehensive search event logging

## Quick Start

1. **Clone and setup**:
   ```bash
   git clone <repository-url>
   cd claude-mcp-websearch-agent
   pip install -r requirements.txt
   ```

2. **Configure environment**:
   ```bash
   cp .env.example .env
   # Edit .env with your API keys
   ```

3. **Start the MCP server**:
   ```bash
   python server.py
   ```

4. **Use the CLI**:
   ```bash
   python cli.py "What's the latest news about AI?"
   ```

## Environment Variables

| Variable | Default | Description |
|----------|---------|-------------|
| `CLAUDE_API_KEY` | - | Your Claude API key (required) |
| `BRAVE_API_KEY` | - | Your Brave Search API key (required) |
| `BRAVE_API_BASE` | `https://api.search.brave.com/res/v1/web/search` | Brave API endpoint |
| `HTTP_TIMEOUT_SECONDS` | `12` | HTTP request timeout |
| `RETRY_ATTEMPTS` | `2` | Number of retry attempts |
| `RETRY_JITTER_MS` | `200` | Jitter for retry delays |
| `ENABLE_ENRICHMENT` | `true` | Enable query enhancement |
| `ENABLE_RERANK` | `true` | Enable result reranking |
| `CACHE_BACKEND` | `sqlite` | Cache backend (sqlite/none) |
| `CACHE_TTL_SECONDS` | `600` | Cache TTL in seconds |
| `MAX_RESULTS` | `8` | Maximum results per search |
| `PORT` | `5001` | MCP server port |
| `MCP_SERVER_URL` | `http://localhost:5001` | MCP server URL |

## Architecture

```
CLI (cli.py) → Claude Client (claude_client.py) → MCP Server (server.py) → Search Bridge (search_bridge.py) → Brave API
```

- **CLI**: Command-line interface for direct queries
- **Claude Client**: Handles Claude API communication and tool calls
- **MCP Server**: Flask-based server exposing the `fetch_web_content` tool
- **Search Bridge**: Core search logic with enrichment, reranking, and caching
- **Brave API**: Web search provider

## API Endpoints

- `GET /health` - Health check
- `GET /` - Server information
- `POST /tool_call` - Handle Claude tool calls

## Tool Schema

The `fetch_web_content` tool accepts:
```json
{
  "query": "string - The search query or topic to look up"
}
```

And returns:
```json
{
  "results": [
    {
      "title": "string",
      "url": "string", 
      "description": "string"
    }
  ]
}
```

## Development

Run tests:
```bash
pytest
```

## License

[Add your license here]