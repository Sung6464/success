import logging
from typing import List, Dict, Optional
from langchain_openai import OpenAIEmbeddings, AzureOpenAIEmbeddings
from langchain_text_splitters import RecursiveCharacterTextSplitter

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


class EmbeddingGenerator:
    """
    Handles chunking and embedding for text documents using OpenAI.
    Supports both standard OpenAI and Azure OpenAI.
    """

    def __init__(
        self,
        api_key: str,
        model: str = "text-embedding-3-small",
        use_azure: bool = False,
        azure_endpoint: Optional[str] = None,
        azure_deployment: Optional[str] = None,
        api_version: Optional[str] = None,
        chunk_size: int = 1000,
        chunk_overlap: int = 200,
    ):
        """
        Initialize embedding generator and text splitter.

        Args:
            api_key: OpenAI API key
            model: Embedding model name (default: text-embedding-3-small)
            use_azure: Whether to use Azure OpenAI
            azure_endpoint: Azure OpenAI endpoint (if using Azure)
            azure_deployment: Azure deployment name (if using Azure)
            api_version: Azure OpenAI API version (if using Azure)
            chunk_size: Text chunk size for splitting
            chunk_overlap: Overlap between chunks
        """
        self.model = model
        self.use_azure = use_azure

        if use_azure and azure_endpoint:
            self.embeddings = AzureOpenAIEmbeddings(
                azure_endpoint=azure_endpoint,
                api_key=api_key,
                azure_deployment=azure_deployment or model,
                api_version=api_version or "2024-02-01"
            )
            logger.info(f"  Initialized Azure OpenAI Embeddings: {azure_deployment or model}")
            logger.info(f"   Endpoint: {azure_endpoint}")
            logger.info(f"   API Version: {api_version or '2024-02-01'}")
        else:
            self.embeddings = OpenAIEmbeddings(
                api_key=api_key,
                model=model
            )
            logger.info(f"  Initialized OpenAI Embeddings: {model}")

        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            length_function=len,
            separators=["\n\n", "\n", ". ", " ", ""]
        )
        logger.info(f"  Text splitter initialized: chunk_size={chunk_size}, chunk_overlap={chunk_overlap}")

    def chunk_and_embed_document(self, doc: Dict) -> Dict:
        """
        Chunk a document and generate embeddings for each chunk.
        Returns a dict with chunk_texts, embeddings, and metadata_list.

        Args:
            doc: Document dict with 'content' and 'metadata'

        Returns:
            {
                'chunk_texts': List[str],
                'embeddings': List[List[float]],
                'metadata_list': List[Dict]
            }
        """
        text = doc.get('content') or doc.get('text')
        if not text:
            logger.warning("No content found in document for chunking.")
            return {'chunk_texts': [], 'embeddings': [], 'metadata_list': []}

        # 1. Split into chunks
        chunks = self.text_splitter.split_text(text)
        if not chunks:
            logger.warning("No chunks generated from document.")
            return {'chunk_texts': [], 'embeddings': [], 'metadata_list': []}

        # 2. Prepare metadata for each chunk
        metadata_list = []
        for idx, _ in enumerate(chunks):
            chunk_metadata = doc['metadata'].copy()
            chunk_metadata['chunk_index'] = idx
            chunk_metadata['total_chunks'] = len(chunks)
            metadata_list.append(chunk_metadata)

        # 3. Generate embeddings for all chunks
        embeddings = self.embed_documents(chunks)

        logger.info(f"  Chunked and embedded document '{doc.get('name', '')}': {len(chunks)} chunks")
        return {
            'chunk_texts': chunks,
            'embeddings': embeddings,
            'metadata_list': metadata_list
        }

    def embed_documents(self, texts: List[str]) -> List[List[float]]:
        """
        Generate embeddings for multiple documents.

        Args:
            texts: List of text strings to embed

        Returns:
            List of embedding vectors
        """
        try:
            embeddings = self.embeddings.embed_documents(texts)
            logger.info(f"  Generated {len(embeddings)} embeddings")
            return embeddings
        except Exception as e:
            logger.error(f"  Error generating embeddings: {e}")
            raise

    def embed_query(self, text: str) -> List[float]:
        """
        Generate embedding for a single query.

        Args:
            text: Query text

        Returns:
            Embedding vector
        """
        try:
            embedding = self.embeddings.embed_query(text)
            return embedding
        except Exception as e:
            logger.error(f"  Error generating query embedding: {e}")
            raise

    def embed_documents_batch(self, texts: List[str], batch_size: int = 100) -> List[List[float]]:
        """
        Generate embeddings in batches to handle rate limits.

        Args:
            texts: List of text strings to embed
            batch_size: Number of texts to process at once

        Returns:
            List of embedding vectors
        """
        all_embeddings = []

        for i in range(0, len(texts), batch_size):
            batch = texts[i:i + batch_size]
            embeddings = self.embed_documents(batch)
            all_embeddings.extend(embeddings)
            logger.info(f"  Processed {min(i + batch_size, len(texts))}/{len(texts)} texts")

        return all_embeddings

    def get_dimension(self) -> int:
        """
        Get the dimension of the embedding vectors.

        Returns:
            Embedding dimension
        """
        # Model dimensions
        dimensions = {
            "text-embedding-3-small": 1536,
            "text-embedding-3-large": 3072,
            "text-embedding-ada-002": 1536,
        }

        return dimensions.get(self.model, 3072)