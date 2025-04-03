from typing import List, Dict, Any
import os
from openai import OpenAI
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import logging

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            # Initialize OpenAI client with explicit API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            self.client = OpenAI(api_key=api_key)
            
            # Initialize Qdrant client
            self.qdrant_client = QdrantClient(
                url=os.getenv("QDRANT_URL", "http://localhost:6333"),
                prefer_grpc=True
            )
            self.collection_name = os.getenv("QDRANT_COLLECTION_NAME", "documents")
            self._ensure_collection_exists()
        except Exception as e:
            logger.error(f"Failed to initialize VectorStore: {str(e)}")
            raise

    def _ensure_collection_exists(self) -> None:
        """Ensure the Qdrant collection exists."""
        try:
            self.qdrant_client.get_collection(self.collection_name)
            logger.info(f"Collection {self.collection_name} already exists")
        except Exception:
            try:
                self.qdrant_client.create_collection(
                    collection_name=self.collection_name,
                    vectors_config=models.VectorParams(
                        size=1536,  # OpenAI embedding dimension
                        distance=models.Distance.COSINE
                    )
                )
                logger.info(f"Created new collection {self.collection_name}")
            except Exception as e:
                logger.error(f"Failed to create collection: {str(e)}")
                raise

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector store with their embeddings.
        
        Args:
            documents: List of document texts to add
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        try:
            # Generate embeddings for all documents
            embeddings = []
            for doc in documents:
                response = self.client.embeddings.create(
                    model="text-embedding-ada-002",
                    input=doc
                )
                embeddings.append(response.data[0].embedding)
            
            # Prepare points for Qdrant
            points = []
            for idx, (doc, embedding) in enumerate(zip(documents, embeddings)):
                point = models.PointStruct(
                    id=idx,
                    vector=embedding,
                    payload={"text": doc}
                )
                points.append(point)
            
            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points
            )
            logger.info(f"Successfully added {len(documents)} documents to vector store")
        except Exception as e:
            logger.error(f"Failed to add documents: {str(e)}")
            raise

    def similarity_search(self, query: str, k: int = 5) -> List[str]:
        """
        Search for similar documents to the query.
        
        Args:
            query: The search query
            k: Number of results to return
            
        Returns:
            List of similar document texts
        """
        try:
            # Generate embedding for the query
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=query
            )
            query_embedding = response.data[0].embedding
            
            # Search in Qdrant
            search_result = self.qdrant_client.search(
                collection_name=self.collection_name,
                query_vector=query_embedding,
                limit=k
            )
            
            # Extract text from results
            return [hit.payload["text"] for hit in search_result]
        except Exception as e:
            logger.error(f"Failed to perform similarity search: {str(e)}")
            raise

    def get_question_embedding(self, question: str) -> List[float]:
        """
        Generate embedding for a question.
        
        Args:
            question: The question text
            
        Returns:
            List of floats representing the embedding
        """
        try:
            response = self.client.embeddings.create(
                model="text-embedding-ada-002",
                input=question
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate question embedding: {str(e)}")
            raise 