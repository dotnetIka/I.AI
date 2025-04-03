from typing import List, Dict, Any
import os
import openai
from pydantic import BaseModel, Field
from dotenv import load_dotenv
import logging

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
            
            logger.info("OpenAIService initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize OpenAIService: {str(e)}")
            raise

    async def answer_question(self, question: str, context: List[str]) -> Dict[str, Any]:
        """
        Answer a question based on the provided context.
        
        Args:
            question: The question to answer
            context: List of relevant context documents
            
        Returns:
            Dict containing the answer and confidence score
        """
        if not context:
            logger.warning("No context provided for question answering")
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
            
            return {
                "answer": parsed_answer.answer,
                "confidence": parsed_answer.confidence
            }
        except Exception as e:
            logger.error(f"Failed to answer question: {str(e)}")
            raise 