from langchain.embeddings.base import Embeddings
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.vectorstores import Chroma
from sentence_transformers import SentenceTransformer
from deep_translator import GoogleTranslator
from langdetect import detect
import os
from typing import Optional
import chromadb
import chromadb.utils.embedding_functions as embedding_functions
from langchain.schema import Document
import boto3
import uuid


Amazon_translation_client=boto3.client(
    service_name='translate',
    region_name='ap-south-1',
    aws_access_key_id='AWS_ACCESS_KEY',
    aws_secret_access_key='AWS_SECRET_KEY'
)

client = chromadb.CloudClient(
  api_key='CHROMA_API_KEY',
  tenant='TENANT',
  database='EchoedAI'
)

embedding_model = SentenceTransformer("sentence-transformers/LaBSE")

class LocalEmbeddings(Embeddings):
    

    def embed_documents(self, texts):
        return embedding_model.encode(texts, convert_to_numpy=True)

    def embed_query(self, text):
        return embedding_model.encode([text], convert_to_numpy=True)
    


local_embeddings = LocalEmbeddings()


class LocalEmbeddingsWrapper:
    
    def __init__(self, embedding_obj: LocalEmbeddings):
        self.model = embedding_obj

    def __call__(self, input):
        if isinstance(input, str):
            return self.model.embed_query(input)
        elif isinstance(input, list):
            return self.model.embed_documents(input)
        else:
            raise ValueError("Input must be str or list[str]")

    def embed_query(self, input):
        return self.__call__(input)

    def name(self):
        return "MyEmbeddingFunc"
    
    

embeddings = LocalEmbeddingsWrapper(local_embeddings)


def _normalize_docs(documents, file_path):
    """
    Ensure all docs are in English:
    - Auto-detect language (using langdetect)
    - Translate to English with Amazon Translate if needed
    """
    for doc in documents:
        try:
            snippet = doc.page_content[:200]
            lang = detect(snippet)            
            if lang != "en":
                response = Amazon_translation_client.translate_text(
                    Text=doc.page_content,
                    SourceLanguageCode=lang,  
                    TargetLanguageCode="en"
                )
                doc.page_content = response["TranslatedText"]
        except Exception as e:
            print(f"⚠️ Normalization failed for {file_path}: {e}")
    return documents

def _normalize_text(doc):
    """
    Ensure all docs are in English:
    - Auto-detect language
    - Translate to English if not already English
    """
    try:
        snippet = doc.page_content[:200]
        lang = detect(snippet)
        if lang != "en":
            response = Amazon_translation_client.translate_text(
                Text=doc.page_content,
                SourceLanguageCode=lang,
                TargetLanguageCode="en"
            )
            doc.page_content = response["TranslatedText"]
    except Exception as e:
        print(f"⚠️ Normalization failed : {e}")
    return doc

'''def ingest_text(file_path):
    loader = TextLoader(file_path, encoding="utf-8")
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    documents = _normalize_docs(documents, file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)


def ingest_pdf(file_path):
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    for doc in documents:
        doc.metadata["source"] = os.path.basename(file_path)

    documents = _normalize_docs(documents, file_path)

    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)
    return FAISS.from_documents(chunks, embeddings)'''

def ingest_text_for_college(
        title: str,
        text: str,
        college_id: str,
        k: int=5
):
    doc_id=f"{college_id}:{title}"
    documents=Document(
        page_content=text,
        metadata={
            "source":title,
            "college_id":college_id,
            "doc_id":doc_id
        }
    )

    documents = _normalize_text(documents)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents([documents])

    collection_name=f"college_{college_id}"

    collection=client.get_or_create_collection(
        name=collection_name,
        embedding_function=embeddings,
        metadata={"description":f"Collection for college {college_id}"}
    )

    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata["doc_id"] for chunk in chunks]
    )
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client   
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return collection,retriever
    

def ingest_pdf_for_colleges(
    file_path: str,
    college_id: str,
    k: int = 5,  
):
    """
    Ingest a PDF into a persistent Chroma DB dedicated to a single college.
    Adds unique metadata so you can later query or delete by college or file.
    """
    loader = PyPDFLoader(file_path)
    documents = loader.load()
    
    for doc in documents:
        doc.metadata.update({
            "source": os.path.basename(file_path),
            "college_id": college_id,
            "doc_id": f"{college_id}:{os.path.basename(file_path)}:{doc.metadata.get('page', 0)}"
        })

    documents = _normalize_docs(documents, file_path)
    splitter = RecursiveCharacterTextSplitter(chunk_size=800, chunk_overlap=100)
    chunks = splitter.split_documents(documents)

    collection_name=f"college_{college_id}"

    
    collection=client.get_or_create_collection(
        name=collection_name,
        embedding_function=embeddings,
        metadata={"description":f"Collection for college {college_id}"}
    )

    collection.delete(where={"source": os.path.basename(file_path)})


    collection.add(
        documents=[chunk.page_content for chunk in chunks],
        metadatas=[chunk.metadata for chunk in chunks],
        ids=[chunk.metadata["doc_id"] for chunk in chunks]
    )
    
    vectorstore = Chroma(
        collection_name=collection_name,
        embedding_function=embeddings,
        client=client   
    )

    retriever = vectorstore.as_retriever(search_kwargs={"k": 5})
    return collection,retriever

    
def delete_pdf_or_text_for_college(
    college_id: str,
    pdf_name: str,
    
) -> Optional[int]:
    collection_name=f"college_{college_id}"

    collection=client.get_collection(collection_name)

    collection.delete(where={"source":pdf_name})

# -----------------------------
# Auto-ingest Folder
# -----------------------------
'''def auto_ingest_data_folder(folder_path="data"):
    """Ingest all PDFs/TXT files in a folder into one FAISS retriever."""
    current_dir = os.path.dirname(os.path.abspath(__file__))
    abs_folder_path = os.path.join(current_dir, folder_path)

    if not os.path.exists(abs_folder_path):
        raise ValueError(f"❌ Folder not found: {abs_folder_path}")

    files = [f for f in os.listdir(abs_folder_path) if f.endswith((".txt", ".pdf"))]
    if not files:
        raise ValueError("❌ No valid .txt or .pdf files found in data folder")

    print(f"📂 Found raw files in {os.path.abspath(abs_folder_path)}: {files}")

    vectorstore = None
    for fname in files:
        fpath = os.path.join(abs_folder_path, fname)
        if fname.endswith(".txt"):
            print(f"➡️ Ingesting TXT: {fname}")
            vs = ingest_text(fpath)
        elif fname.endswith(".pdf"):
            print(f"➡️ Ingesting PDF: {fname}")
            vs = ingest_pdf(fpath)
        else:
            continue

        if vectorstore is None:
            vectorstore = vs
        else:
            vectorstore.merge_from(vs)

    print(f"✅ Ingested {len(files)} files: {files}")
    return vectorstore.as_retriever(search_kwargs={"k": 5})'''
