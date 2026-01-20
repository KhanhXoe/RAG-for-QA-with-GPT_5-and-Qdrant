import requests
import json

# API endpoint
BASE_URL = "http://localhost:8000"

# Test documents
documents = {
    "documents": [
        {
            "text": "The Python programming language was created by Guido van Rossum and was released in 1991. It emphasizes code readability with its notable use of significant indentation.",
            "metadata": {"source": "wiki", "topic": "programming"}
        },
        {
            "text": "Python is dynamically typed and garbage-collected. It supports multiple programming paradigms, including structured, object-oriented, and functional programming.",
            "metadata": {"source": "wiki", "topic": "programming"}
        },
        {
            "text": "OpenAI was founded in 2015. Initially a non-profit organization, it transitioned to a 'capped-profit' model in 2019.",
            "metadata": {"source": "news", "topic": "AI"}
        }
    ]
}

# Add documents
print("\nAdding documents...")
response = requests.post(f"{BASE_URL}/documents", json=documents)
print(f"Status: {response.status_code}")
print(json.dumps(response.json(), indent=2))

# Test queries
test_queries = [
    "When was Python created and by whom?",
    "What type of programming language is Python?",
    "Tell me about OpenAI's founding",
    "What is the capital of France?"  # This should fail due to no relevant context
]

print("\nTesting queries...")
for query in test_queries:
    print(f"\nQuery: {query}")
    response = requests.post(f"{BASE_URL}/query", json={"query": query})
    print(f"Status: {response.status_code}")
    if response.status_code == 200:
        print(json.dumps(response.json(), indent=2))
    else:
        print(f"Error: {response.text}")