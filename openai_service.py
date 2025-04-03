from typing import List, Dict, Any
import os
import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging
from cachetools import TTLCache

load_dotenv()

logger = logging.getLogger(__name__)

class Answer(BaseModel):
    answer: str = Field(description="The answer to the question")
    confidence: float = Field(description="Confidence score between 0 and 1")

class OpenAIService:
    def __init__(self):
        try:
            # Get API key
            api_key = os.getenv("OPENAI_API_KEY")
            if not api_key:
                raise ValueError("OPENAI_API_KEY environment variable is not set")
            
            # Set the API key for the openai module
            openai.api_key = api_key
            
            # Initialize question cache with 1-hour TTL and max size
            self.question_cache = TTLCache(maxsize=1024, ttl=3600) # 3600 seconds = 1 hour
            
            logger.info("OpenAIService initialized successfully with 1-hour TTL question cache.")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIService: {str(e)}")
            raise

    async def answer_question(self, question: str, context: List[str]) -> Dict[str, Any]:
        """
        Answer a question based on the provided context.
        Uses an in-memory TTL cache (1 hour) based on the question text.
        
        Args:
            question: The question to answer
            context: List of relevant context documents
            
        Returns:
            Dict containing the answer and confidence score
        """
        normalized_question = question.lower() # Normalize for caching
        
        # Check cache first
        if normalized_question in self.question_cache:
            logger.info(f"Cache hit for question: '{question}'")
            return self.question_cache[normalized_question]
        
        logger.info(f"Cache miss for question: '{question}'")

        if not context:
            logger.warning("No context provided for question answering")
            # Don't cache 'I don't know' answers due to lack of context
            return {
                "answer": "I don't have enough information to answer this question.",
                "confidence": 0.0
            }

        try:
            # Format the prompt
            system_prompt = """You are a helpful assistant that answers questions about the Democratic Republic of Georgia (1918-1921). 
            Use the provided context to answer the question. If you don't know the answer, say 'I don't know'.
            
            Your response should be in JSON format with two fields:
            - answer: The answer to the question
            - confidence: A confidence score between 0 and 1
            
            Example format:
            {
                "answer": "The answer to the question",
                "confidence": 0.95
            }
            """
            
            user_prompt = f"""Context: {' '.join(context)}

            Question: {question}"""
            
            # Log the context being used (each section fully)
            logger.debug(f"--- Context being sent to OpenAI ({len(context)} sections) ---")
            for i, doc_section in enumerate(context):
                logger.debug(f"Context Section [{i+1}/{len(context)}]:\n{doc_section}")
            logger.debug("--- End of Context ---")
            
            # Get completion from OpenAI using the module directly
            response = openai.ChatCompletion.create(
                model="gpt-4o",
                messages=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": user_prompt}
                ],
                temperature=0.7,
                response_format={"type": "json_object"}
            )
            
            # Parse the response
            answer_text = response.choices[0].message.content
            parsed_answer = Answer.model_validate_json(answer_text)
            
            result = {
                "answer": parsed_answer.answer,
                "confidence": parsed_answer.confidence
            }
            
            # Store successful result in cache
            self.question_cache[normalized_question] = result
            logger.info(f"Stored answer in cache for question: '{question}'")
            
            return result
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise 