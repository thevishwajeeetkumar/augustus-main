# Search Tool Update

This document explains the adjustments made when removing the DuckDuckGo search tool
from the Augustus agent. It summarizes the previous behaviour and the new structure
so future contributors can understand the impact of the change.

---

## Previous Structure

- **Agent tools**:  
  - `youtube_rag_tool` for retrieving context from the user’s Pinecone index.  
  - `DuckDuckGoSearchRun` for open web lookups when the agent determined external
    search was necessary.
- **Dependencies**:  
  - `duckduckgo-search` and its transitive packages (`ddgs`, `curl-cffi`, etc.) were
    installed via `requirements.txt`.
- **Architecture**:  
  - Documentation highlighted an additional dependency on DuckDuckGo as part of the
    tool stack.
- **Runtime considerations**:  
  - Outbound search requests relied on third-party infrastructure and introduced
    another point of failure/network variability.

---

## New Structure

- **Agent tools**:  
  - `youtube_rag_tool` remains the primary retrieval path.
  - `google_search_tool` provides curated Google Custom Search results when the agent
    needs supplemental context beyond the video knowledge base.
- **Dependencies**:  
  - `duckduckgo-search` and `ddgs` have been removed.
  - `requests` is explicitly listed to support the Google Custom Search integration.
- **Architecture**:  
  - Documentation now reflects FastAPI orchestrating OpenAI, Pinecone, and Google
    Custom Search.
- **Operational impact**:  
  - Restores optional web augmentation while using an officially supported API.
  - Keeps the failure surface controlled with structured responses from Google.

---

## Rationale

The initial DuckDuckGo integration provided general web search but introduced
instability and dependency sprawl. Temporarily operating on YouTube-only RAG proved
predictable, yet certain queries benefit from fresh, external context. Google Custom
Search offers a structured, well-supported API that complements the YouTube pipeline
without reintroducing the previous fragility.

---

## Follow-Up Work

- Monitor usage to ensure the Google Custom Search quotas and latency meet SLAs.
- Consider a configuration flag to disable/enable web search per deployment.
- Explore richer summarisation of Google results (e.g., citations, metadata scoring).


