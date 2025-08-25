import os
import re
import json
import requests
from typing import Dict, List, Any, Optional, Literal, Protocol
from dataclasses import dataclass, asdict
import anthropic

# Configuration from environment variables
BRAVE_API_BASE = os.environ.get("BRAVE_API_BASE", "https://api.search.brave.com/res/v1/web/search")
BRAVE_API_KEY = os.environ.get("BRAVE_API_KEY", "")
HTTP_TIMEOUT_SECONDS = int(os.environ.get("HTTP_TIMEOUT_SECONDS", "12"))
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

class BraveSearchProvider:
    def __init__(self, api_base: str = BRAVE_API_BASE, api_key: str = BRAVE_API_KEY):
        self.api_base = api_base
        self.api_key = api_key
        self.timeout = HTTP_TIMEOUT_SECONDS

    def search(self, query: str, count: int = 10) -> List[WebResult]:
        if not self.api_key:
            print("Missing BRAVE_API_KEY. Set it in your environment.")
            return []

        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": self.api_key
        }
        
        params = {
            "q": query,
            "count": min(count, 10)  # Brave API limit
        }

        try:
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
        except requests.exceptions.Timeout:
            print(f"Search request timed out after {self.timeout} seconds")
            return []
        except requests.exceptions.RequestException as e:
            print(f"Search provider request failed: {e}")
            return []
        except Exception as e:
            print(f"Unexpected error during search: {e}")
            return []

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
    

        
    