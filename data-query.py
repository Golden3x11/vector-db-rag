import argparse

from elasticsearch import Elasticsearch
from sentence_transformers import SentenceTransformer


def main(query_text):
    if not query_text:
        query_text = input_with_default("Enter query text", "python programming language")

    es = get_es_client()
    model = SentenceTransformer('all-MiniLM-L6-v2')

    documents = fetch_documents(es, model, query_text)

    print("\nTop 5 documents:")
    for idx, doc in enumerate(documents, 1):
        print(f"{idx}. {doc}")


def get_es_client(verbose=True):
    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", "changeme"),
        request_timeout=180
    )
    if verbose:
        print(es.info())

    return es

def input_with_default(prompt, default=''):
    try:
        user_input = input(f"{prompt} (default: {default}): ")
        return user_input if user_input else default
    except EOFError:
        return default

def fetch_documents(es, model, query_text, index_name="medium_articles"):
    query_vector = model.encode(query_text).tolist()

    response = es.search(
        index=index_name,
        knn={
            "field": "text_vector",
            "query_vector": query_vector,
            "k": 5,
            "num_candidates": 50
        }
    )

    return [hit['_source']['text'] for hit in response['hits']['hits']]

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search documents using Elasticsearch and Sentence Transformers")
    parser.add_argument('--query_text', type=str, help="Query text to search for")
    args = parser.parse_args()

    main(args.query_text)