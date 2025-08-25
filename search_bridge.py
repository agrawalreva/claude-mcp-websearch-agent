import os
import re
import json
import requests
import time
import random
from typing import Dict, List, Any, Optional, Literal, Protocol
from dataclasses import dataclass, asdict
import anthropic

# Configuration from environment variables
BRAVE_API_BASE = os.environ.get("BRAVE_API_BASE", "https://api.search.brave.com/res/v1/web/search")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "12"))
RETRY_ATTEMPTS = int(os.environ.get("RETRY_ATTEMPTS", "2"))
RETRY_JITTER_MS = int(os.environ.get("RETRY_JITTER_MS", "200"))
ENABLE_ENRICHMENT = os.environ.get("ENABLE_ENRICHMENT", "true").lower() == "true"
CLAUDE_API_KEY = os.environ.get("CLAUDE_API_KEY", "")

LLMProvider = Literal["claude"]

@dataclass
class WebResult:
    title: str
    url: str
    description: str

class SearchProvider(Protocol):
    def search(self, query: str, count: int = 10) -> List[WebResult]:
        ...

def log_search_event(level: str, message: str, **kwargs):
    """Simple structured logging for search events."""
    log_data = {
        "timestamp": time.time(),
        "level": level,
        "message": message,
        **kwargs
    }
    print(f"[{level.upper()}] {message} | " + " | ".join(f"{k}={v}" for k, v in kwargs.items()))

def enrich_query(query: str) -> List[str]:
    """Apply simple query enrichment: typo fixes and synonyms."""
    if not ENABLE_ENRICHMENT:
        return [query]
    
    enriched = [query]
    
    # Simple typo corrections
    typos = {
        "teh": "the",
        "recieve": "receive",
        "seperate": "separate",
        "occured": "occurred",
        "neccessary": "necessary"
    }
    
    for typo, correction in typos.items():
        if typo in query.lower():
            corrected = query.lower().replace(typo, correction)
            if corrected not in enriched:
                enriched.append(corrected)
    
    # Simple synonym expansion (limited to prevent explosion)
    synonyms = {
        "fast": ["quick", "rapid"],
        "big": ["large", "huge"],
        "good": ["great", "excellent"],
        "bad": ["poor", "terrible"]
    }
    
    for word, syns in synonyms.items():
        if word in query.lower() and len(enriched) < 3:  # Cap at 3 variants
            for syn in syns:
                if len(enriched) < 3:
                    variant = query.lower().replace(word, syn)
                    if variant not in enriched:
                        enriched.append(variant)
    
    return enriched[:3]  # Ensure we don't exceed 3 variants

class BraveSearchProvider:
    def __init__(self, api_base: str = BRAVE_API_BASE, api_key: str = BRAVE_API_KEY):
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = HTTP_TIMEOUT_SECONDS
        self.max_retries = RETRY_ATTEMPTS
        self.jitter_ms = RETRY_JITTER_MS

    def search(self, query: str, count: int = 10) -> List[WebResult]:
        if not self.api_key:
            log_search_event("error", "Missing API key", engine="brave")
            return []

        start_time = time.time()
        log_search_event("info", "Starting search", engine="brave", query=query, max_results=count)

        # Apply query enrichment
        enriched_queries = enrich_query(query)
        if len(enriched_queries) > 1:
            log_search_event("info", "Query enriched", original=query, variants=len(enriched_queries))

        all_results = []
        for enriched_query in enriched_queries:
            try:
                results = self._perform_search(enriched_query, count)
                all_results.extend(results)
                if len(all_results) >= count:
                    break
            except Exception as e:
                log_search_event("warn", "Enriched query failed", query=enriched_query, error=str(e))
                continue

        # Deduplicate results by URL
        seen_urls = set()
        unique_results = []
        for result in all_results:
            if result.url not in seen_urls:
                seen_urls.add(result.url)
                unique_results.append(result)
                if len(unique_results) >= count:
                    break

        latency_ms = int((time.time() - start_time) * 1000)
        log_search_event("info", "Search completed", engine="brave", query=query, 
                       results_count=len(unique_results), latency_ms=latency_ms, 
                       enrichment_enabled=ENABLE_ENRICHMENT)
        
        return unique_results[:count]

    def _perform_search(self, query: str, count: int) -> List[WebResult]:
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 10)  # Brave API limit
        }

        response = requests.get(
            self.api_base,
            headers=headers,
            params=params,
            timeout=self.timeout
        )

        response.raise_for_status()
        data = response.json()
        results = []

        if "web" in data and "results" in data["web"]:
            for item in data["web"]["results"][:count]:
                results.append(WebResult(
                    title=item.get("title", ""),
                    url=item.get("url", ""),
                    description=item.get("description", "")
                ))
        
        return results

    def _should_retry(self, error: Exception) -> bool:
        """Determine if the error is retryable."""
        if hasattr(error, 'response') and error.response is not None:
            status_code = error.response.status_code
            return status_code >= 500 or status_code == 429
        return False

    def _wait_with_jitter(self, attempt: int):
        """Wait with exponential backoff and jitter."""
        base_delay = 2 ** attempt
        jitter = random.randint(0, self.jitter_ms) / 1000.0
        delay = base_delay + jitter
        time.sleep(delay)

def get_search_provider() -> SearchProvider:
    """Factory function to get the configured search provider."""
    return BraveSearchProvider()

class ClaudeMCPBridge:
    def __init__(self, llm_provider: LLMProvider = "claude"):
        self.search_provider = get_search_provider()
        self.llm_provider = llm_provider

        if llm_provider == "claude":
            if not CLAUDE_API_KEY:
                raise ValueError("Missing CLAUDE_API_KEY. Set it in your environment.")
            self.claude_client = anthropic.Anthropic(api_key=CLAUDE_API_KEY)

    def extract_website_queries_with_llm(self, user_message: str) -> List[str]:
        if self.llm_provider == "claude":
            return self._extract_with_claude(user_message)
        else:
            return ["error"]
        
    def _extract_with_claude(self, user_message: str) -> List[str]:
        try:
            response = self.claude_client.messages.create(
                model="claude-3-sonnet-20240229",
                max_tokens=1000,
                temperature=0.1,
                system="You are a helpful assistant that identifies web search queries in user message. Extract any specific website or topic queries the user wants information about. Return results as a JSON object with a 'queries' field containing an array of strings. If no queries are found, return an empty array.",
                messages=[
                    {"role": "user", "content": user_message}
                ]
            )

            content = response.content[0].text
            json_match = re.search(r'```(?:json)?\s*({.*?})\s*```', content, re.DOTALL)
            if json_match:
                result = json.loads(json_match.group(1))
            else:
                try:
                    result = json.loads(content)
                except:
                    return []
            queries = result.get("queries", [])
            return queries
        
        except Exception as e:
            print(f"Error extracting queries with Claude: {e}")
            return []
        
def handle_claude_tool_call(tool_params: Dict[str, Any]) -> Dict[str, Any]:
    query = tool_params.get("query", "")
    if not query:
        return {"error": "Missing query parameter"}
    
    bridge = ClaudeMCPBridge()
    results = bridge.search_provider.search(query)

    return {
        "results": [asdict(result) for result in results]
    }
    

        
    