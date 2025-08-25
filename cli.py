import sys
import os
import requests
import argparse
import json
from claude_client import ClaudeClient

def check_mcp_server():
    mcp_url = os.environ.get("MCP_SERVER_URL", "http://localhost:5001")
    try:
        response = requests.get(f"{mcp_url}/health", timeout=2)
        if response.status_code == 200:
            return True
        return False
    except requests.exceptions.RequestException:
        return False

def main():
    parser = argparse.ArgumentParser(description="Claude web search interface with MCP integration")
    parser.add_argument("query", nargs="*", help="The question to ask Claude")
    args = parser.parse_args()

    if not os.environ.get("CLAUDE_API_KEY"):
        print("Missing CLAUDE_API_KEY. Set it in your environment.")
        sys.exit(1)

    if args.query:
        query = " ".join(args.query)
    else:
        query = input("Enter your question: ")
    
    client = ClaudeClient()

    print(f"Searching: {query}")

    try:
        answer = client.get_final_answer(query)
        print("Answer:", answer)
    except Exception as e:
        print(f"Error: {e}")

if __name__ == "__main__":
    main()

    
