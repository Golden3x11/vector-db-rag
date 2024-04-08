import pandas as pd
from elasticsearch import Elasticsearch
from langchain_core.documents import Document
from tqdm import tqdm
from langchain_text_splitters import CharacterTextSplitter
from sentence_transformers import SentenceTransformer

INDEX_NAME = "medium_articles"
INDEX_SETTINGS = {
    "settings": {
        "number_of_shards": 5,  # it is also default value
        "number_of_replicas": 0
    },
    "mappings": {
        "properties": {
            "text": {"type": "text"},
            "text_vector": {
                "type": "dense_vector",
                "dims": 384,
                "index": "true",
                "similarity": "cosine" # cosine similarity
            },
        }
    }
}
VECTORIZER_MODEL = 'all-MiniLM-L6-v2'



def setup_elasticsearch():
    es = Elasticsearch(
        "http://localhost:9200",
        basic_auth=("elastic", "changeme"),
        request_timeout=180
    )
    if not es.indices.exists(index=INDEX_NAME):
        es.indices.create(index=INDEX_NAME, body=INDEX_SETTINGS)
        print(f"Index {INDEX_NAME} created")
    return es


def split_text(text):
    text_splitter = CharacterTextSplitter(chunk_size=300, chunk_overlap=50)
    return text_splitter.split_documents([Document(text)])


def vectorize_text(text, model):
    return model.encode(text).tolist()


def index_document(es, doc, doc_id):
    es.index(index=INDEX_NAME, body=doc, id=doc_id)


def main():
    model = SentenceTransformer(VECTORIZER_MODEL)
    es = setup_elasticsearch()
    df = pd.read_csv('./data/medium.csv')

    for idx, row in tqdm(df.iterrows(), "Indexing documents", total=len(df)):
        chunks = split_text(row['Text'])
        for chunk_idx, chunk_text in enumerate(chunks):
            chunk_text = chunk_text.page_content
            chunk_id = f"{row['Title']} - Row {idx} - Chunk {chunk_idx}"
            text_vector = vectorize_text(chunk_text, model)
            doc = {
                "text": chunk_text,
                "text_vector": text_vector
            }
            index_document(es, doc, chunk_id)


if __name__ == "__main__":
    main()