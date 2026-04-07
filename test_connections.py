# test_connections.py
# run this once manually: python test_connections.py
# not part of the test suite — just a sanity check before building anything

import os
import sys
from dotenv import load_dotenv

load_dotenv()

results = {}


def check_groq():
    print("Checking Groq...")
    try:
        from langchain_groq import ChatGroq
        from config.settings import GROQ_MODEL

        llm = ChatGroq(
            api_key=os.getenv("GROQ_API_KEY"),
            model=GROQ_MODEL,
            max_tokens=50,
        )
        response = llm.invoke("Say the word hello.")
        print(f"  ok — got: {response.content[:40]}")
        results["groq"] = True
    except Exception as e:
        print(f"  failed — {e}")
        results["groq"] = False


def check_cohere():
    print("Checking Cohere...")
    try:
        import cohere
        from config.settings import COHERE_MODEL

        client = cohere.ClientV2(api_key=os.getenv("COHERE_API_KEY"))
        response = client.rerank(
            model=COHERE_MODEL,
            query="test query",
            documents=["first document", "second document"],
            top_n=2,
        )
        print(f"  ok — reranked {len(response.results)} docs")
        results["cohere"] = True
    except Exception as e:
        print(f"  failed — {e}")
        results["cohere"] = False


def check_tavily():
    print("Checking Tavily...")
    try:
        from tavily import TavilyClient

        client = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))
        response = client.search("what is LangGraph", max_results=2)
        count = len(response.get("results", []))
        print(f"  ok — got {count} results")
        results["tavily"] = True
    except Exception as e:
        print(f"  failed — {e}")
        results["tavily"] = False


def check_langfuse():
    print("Checking Langfuse...")
    try:
        from langfuse import Langfuse

        client = Langfuse(
            public_key=os.getenv("LANGFUSE_PUBLIC_KEY"),
            secret_key=os.getenv("LANGFUSE_SECRET_KEY"),
            host=os.getenv("LANGFUSE_HOST", "https://cloud.langfuse.com"),
        )
        # .trace() was removed in newer versions, auth_check() is the replacement
        client.auth_check()
        print("  ok — credentials verified")
        results["langfuse"] = True
    except Exception as e:
        print(f"  failed — {e}")
        results["langfuse"] = False


def check_embeddings():
    # no API call here, just making sure the package installed correctly
    print("Checking embeddings (local)...")
    try:
        from sentence_transformers import SentenceTransformer
        from config.settings import EMBEDDING_MODEL, EMBEDDING_DEVICE

        model = SentenceTransformer(EMBEDDING_MODEL, device=EMBEDDING_DEVICE)
        vec = model.encode("test sentence")
        print(f"  ok — dimension: {len(vec)}")
        results["embeddings"] = True
    except Exception as e:
        print(f"  failed — {e}")
        results["embeddings"] = False


def print_summary():
    print("\n" + "-" * 40)
    all_passed = True
    for service, passed in results.items():
        status = "PASS" if passed else "FAIL"
        print(f"  {service:<15} {status}")
        if not passed:
            all_passed = False
    print("-" * 40)

    if all_passed:
        print("everything is connected, ready for day 2\n")
    else:
        print("fix the failures above before moving on\n")

    return all_passed


if __name__ == "__main__":
    check_groq()
    check_cohere()
    check_tavily()
    check_langfuse()
    check_embeddings()

    passed = print_summary()
    sys.exit(0 if passed else 1)