import os
import pytest
from unittest.mock import Mock, patch
from search_bridge import WebResult, enrich_query, rerank_results, SearchCache

def test_web_result():
    """Test WebResult dataclass creation."""
    result = WebResult(
        title="Test Title",
        url="https://example.com",
        description="Test description"
    )
    assert result.title == "Test Title"
    assert result.url == "https://example.com"
    assert result.description == "Test description"

def test_enrich_query_disabled():
    """Test query enrichment when disabled."""
    os.environ["ENABLE_ENRICHMENT"] = "false"
    queries = enrich_query("test query")
    assert queries == ["test query"]

def test_enrich_query_typo_correction():
    """Test typo correction in query enrichment."""
    os.environ["ENABLE_ENRICHMENT"] = "true"
    queries = enrich_query("teh test")
    assert "the test" in queries
    assert "teh test" in queries

def test_enrich_query_synonym_expansion():
    """Test synonym expansion in query enrichment."""
    os.environ["ENABLE_ENRICHMENT"] = "true"
    queries = enrich_query("fast car")
    assert "fast car" in queries
    assert "quick car" in queries or "rapid car" in queries

def test_rerank_results_disabled():
    """Test reranking when disabled."""
    os.environ["ENABLE_RERANK"] = "false"
    results = [
        WebResult("Title 1", "https://1.com", "Description 1"),
        WebResult("Title 2", "https://2.com", "Description 2")
    ]
    reranked = rerank_results("test", results)
    assert len(reranked) == 2

def test_rerank_results_scoring():
    """Test that reranking properly scores and sorts results."""
    os.environ["ENABLE_RERANK"] = "true"
    results = [
        WebResult("Generic Title", "https://1.com", "Generic description"),
        WebResult("Test Title", "https://2.com", "Test description with test word")
    ]
    reranked = rerank_results("test", results)
    # The second result should be ranked higher due to "test" matches
    assert reranked[0].title == "Test Title"

def test_cache_initialization():
    """Test cache initialization."""
    cache = SearchCache(ttl_seconds=300)
    assert cache.ttl_seconds == 300

if __name__ == "__main__":
    pytest.main([__file__]) 