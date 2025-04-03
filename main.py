from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import List
from vector_store import VectorStore
from openai_service import OpenAIService
import os
from dotenv import load_dotenv
import logging
import uvicorn
import time

# Configure logging
logging.basicConfig(
    level=logging.DEBUG,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

# Initialize FastAPI app
app = FastAPI(
    title="AI Question Answering API",
    description="API for answering questions about Georgian history (1918-1921)",
    version="1.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize services
try:
    vector_store = VectorStore()
    openai_service = OpenAIService()
    logger.info("Services initialized successfully")
except Exception as e:
    logger.error(f"Failed to initialize services: {str(e)}")
    raise

class QuestionRequest(BaseModel):
    question: str

class AnswerResponse(BaseModel):
    answer: str
    duration_seconds: float

def load_georgian_history() -> List[str]:
    """
    Load and process the Georgian history text file.
    
    Returns:
        List of document sections
    """
    try:
        with open('data/georgian_history.txt', 'r', encoding='utf-8') as file:
            content = file.read()
            # Split content into sections based on double newlines
            sections = [section.strip() for section in content.split('\n\n') if section.strip()]
            logger.info(f"Loaded {len(sections)} sections from Georgian history")
            return sections
    except Exception as e:
        logger.error(f"Failed to load Georgian history: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error loading Georgian history: {str(e)}"
        )

@app.post("/ask", response_model=AnswerResponse)
async def ask_question(request: QuestionRequest):
    """
    Answer a question about Georgian history.
    
    Args:
        request: QuestionRequest containing the question
        
    Returns:
        AnswerResponse containing the generated answer and duration
    """
    start_time = time.time()
    try:
        # Get similar documents from vector store
        similar_docs = vector_store.similarity_search(request.question)
        logger.info(f"Found {len(similar_docs)} similar documents")
        
        # Generate answer using OpenAI
        answer_data = await openai_service.answer_question(request.question, similar_docs)
        logger.info("Successfully generated answer")
        
        end_time = time.time()
        duration = end_time - start_time
        
        return AnswerResponse(
            answer=answer_data["answer"], 
            duration_seconds=duration
        )
    except Exception as e:
        end_time = time.time()
        duration = end_time - start_time
        logger.error(f"Failed to answer question after {duration:.2f} seconds: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error answering question: {str(e)}"
        )

@app.post("/generate-embeddings")
async def generate_embeddings():
    """
    Generate embeddings for the Georgian history text and store them in Qdrant.
    """
    try:
        # Load and process the Georgian history text
        sections = load_georgian_history()
        
        # Add documents to vector store
        vector_store.add_documents(sections)
        logger.info("Successfully generated and stored embeddings")
        
        return {"message": "Georgian history documents processed and embeddings generated successfully"}
    except Exception as e:
        logger.error(f"Failed to generate embeddings: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Error generating embeddings: {str(e)}"
        )

if __name__ == "__main__":
    uvicorn.run(
        "main:app",
        host="0.0.0.0",
        port=8000,
        reload=True
    ) 