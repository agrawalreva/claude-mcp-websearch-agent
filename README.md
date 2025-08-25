# Integration of Brave API with Claude via MCP for structured retrieval pipelines

This project connects Claude to the Brave Search API through the Model Context Protocol (MCP).  
It gives Claude the ability to pull search results, apply ranking logic, and deliver structured outputs.  

---

## Introduction

Claude can reason over text with precision, but it cannot fetch or verify information on its own.  
This project bridges that gap by exposing external search to Claude through the Model Context Protocol (MCP). Brave was chosen as the provider because its API is stable, responses are consistent and the design respects privacy.  

The system integrates Brave Search with Claude using a retrieval pipeline. Queries are normalized, enhanced for typos and synonyms and executed with configurable HTTP timeouts. Requests are retried with exponential backoff and jitter then cached in SQLite to avoid redundant calls.  

Results go through a reranking step that applies TF-IDF style scoring to surface the most relevant entries. The pipeline runs behind a Flask MCP server that exposes the `fetch_web_content` tool, includes health check endpoints, and records logs for traceability. Configuration is handled through environment variables so the setup stays portable.  

Development support includes a pytests that validates caching, retries and tool responses. The cache layer is abstracted so it can be replaced without touching the rest of the system. The final output is clean JSON with titles, URLs, and descriptions that Claude can reason with directly.  

The result is a retrieval service that allows Claude to operate with context beyond its training data, producing answers anchored in real information.


---

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

---

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

---

## Tech Stack

### Programming Language
- Python 3.10+

### Frameworks and Libraries
- Flask (MCP server and API routing)  
- Requests (HTTP client for Brave API calls)  
- SQLAlchemy (optional cache abstraction)  
- Pytest (unit and integration testing)  

### External APIs
- Brave Search API (search provider)  
- Claude API (LLM integration through MCP)  

### Databases
- SQLite (default cache backend)  

### Protocols
- Model Context Protocol (MCP) for tool integration  