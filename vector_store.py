from typing import List, Dict, Any
import os
import openai
from qdrant_client import QdrantClient
from qdrant_client.http import models
from qdrant_client.http.models import Distance, VectorParams
from dotenv import load_dotenv
import logging
import hashlib

load_dotenv()

logger = logging.getLogger(__name__)

class VectorStore:
    def __init__(self):
        try:
            # Get API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            # Set the API key for the openai module
            openai.api_key = api_key
            
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
        """
        Ensure the Qdrant collection exists.
        """
        try:
            # Try to get the collection first
            try:
                self.qdrant_client.get_collection(self.collection_name)
                logger.info(f"Collection {self.collection_name} already exists")
                return
            except Exception:
                # Collection doesn't exist, create it
                pass

            # Create collection with default parameters
            self.qdrant_client.create_collection(
                collection_name=self.collection_name,
                vectors_config=VectorParams(
                    size=1536,  # OpenAI embedding dimension
                    distance=Distance.COSINE
                )
            )
            logger.info(f"Created new collection {self.collection_name}")
        except Exception as e:
            if "already exists" in str(e):
                logger.info(f"Collection {self.collection_name} already exists")
                return
            logger.error(f"Failed to ensure collection exists: {str(e)}")
            raise

    def add_documents(self, documents: List[str]) -> None:
        """
        Add documents to the vector store with their embeddings.
        Uses content-based hashing for IDs to support idempotent updates.
        
        Args:
            documents: List of document texts to add
        """
        if not documents:
            logger.warning("No documents provided to add_documents")
            return

        try:
            points_to_upsert = []
            # Process documents in batches for embedding generation if needed (optional optimization)
            # For simplicity here, we process one by one
            for doc in documents:
                if not doc.strip(): # Skip empty documents
                    continue
                    
                # Generate embedding
                response = openai.Embedding.create(
                    model="text-embedding-ada-002",
                    input=doc
                )
                embedding = response.data[0].embedding
                
                # Generate content-based ID using SHA-256 hash
                hasher = hashlib.sha256(doc.encode('utf-8'))
                # Convert hex hash to integer and fit into positive 64-bit range
                point_id = int(hasher.hexdigest(), 16) % (2**63)
                
                # Prepare point for Qdrant
                point = models.PointStruct(
                    id=point_id, # Use the hash-based ID
                    vector=embedding,
                    payload={"text": doc}
                )
                points_to_upsert.append(point)
            
            if not points_to_upsert:
                logger.info("No valid documents found to add/update.")
                return

            # Upload points to Qdrant
            self.qdrant_client.upsert(
                collection_name=self.collection_name,
                points=points_to_upsert,
                wait=True # Wait for operation to complete
            )
            logger.info(f"Successfully upserted {len(points_to_upsert)} documents to vector store using content-based IDs")
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
            response = openai.Embedding.create(
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
            response = openai.Embedding.create(
                model="text-embedding-ada-002",
                input=question
            )
            return response.data[0].embedding
        except Exception as e:
            logger.error(f"Failed to generate question embedding: {str(e)}")
            raise 