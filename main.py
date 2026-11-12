from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import StreamingResponse
from pydantic import BaseModel
from typing import List, Optional, Dict, Any
import os
import json
import uuid
from pathlib import Path
import asyncio

# Initialize FastAPI app
app = FastAPI(title="QuestionCraft API", version="1.0.0")

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, specify exact origins
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Pydantic models for request/response validation
class ProcessRequest(BaseModel):
    filename: str

class EmbeddingRequest(BaseModel):
    filename: str

class QuestionRequest(BaseModel):
    question: str

class AnswerSubmission(BaseModel):
    questionIndex: int
    answer: str

class EvaluationRequest(BaseModel):
    topic: str
    answers: List[AnswerSubmission]

# In-memory storage for demo purposes
# In production, you would use a database
uploaded_files = {}
processed_content = {}
topics_data = {}
summary_data = {}
vector_stores = {}

# Helper functions
def get_file_path(filename: str) -> Path:
    """Get the full path for a file"""
    upload_dir = Path("uploads")
    upload_dir.mkdir(exist_ok=True)
    return upload_dir / filename

def save_file_content(filename: str, content: str):
    """Save processed content"""
    processed_content[filename] = content

def get_file_content(filename: str) -> str:
    """Get processed content"""
    return processed_content.get(filename, "")

def save_topics(filename: str, topics: List[str]):
    """Save extracted topics"""
    topics_data[filename] = topics

def get_topics(filename: str) -> List[str]:
    """Get extracted topics"""
    return topics_data.get(filename, [])

def save_summary(filename: str, summary: str):
    """Save document summary"""
    summary_data[filename] = summary

def get_summary(filename: str) -> str:
    """Get document summary"""
    return summary_data.get(filename, "No summary available for this document.")

# API Endpoints

@app.get("/")
async def root():
    return {"message": "QuestionCraft API is running!"}

@app.post("/api/upload")
async def upload_pdf(file: UploadFile = File(...)):
    """Handle PDF file uploads"""
    try:
        # Generate a unique filename to avoid conflicts
        unique_filename = f"{uuid.uuid4()}_{file.filename}"
        
        # Save file to disk
        file_path = get_file_path(unique_filename)
        with open(file_path, "wb") as buffer:
            content = await file.read()
            buffer.write(content)
        
        # Store in memory (in production, you'd store in a database)
        uploaded_files[unique_filename] = {
            "original_name": file.filename,
            "size": len(content),
            "content_type": file.content_type
        }
        
        return {
            "filename": unique_filename,
            "original_name": file.filename,
            "message": "File uploaded successfully"
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to upload file: {str(e)}")

@app.post("/api/process")
async def process_pdf(request: ProcessRequest):
    """Process PDF and extract text"""
    try:
        filename = request.filename
        
        # Check if file exists
        if filename not in uploaded_files:
            raise HTTPException(status_code=404, detail="File not found")
        
        # Simulate PDF processing
        # In a real implementation, you would use pdfplumber or similar
        file_path = get_file_path(filename)
        if not file_path.exists():
            raise HTTPException(status_code=404, detail="File not found on disk")
        
        # Simulate text extraction
        # For demo purposes, we'll create mock content
        mock_content = """
        Artificial Intelligence and Machine Learning
        
        Artificial Intelligence (AI) is transforming industries across the globe. 
        Machine Learning (ML), a subset of AI, enables computers to learn from data 
        without being explicitly programmed.
        
        Key Concepts in Machine Learning:
        1. Supervised Learning - Uses labeled data to train models
        2. Unsupervised Learning - Finds patterns in unlabeled data
        3. Reinforcement Learning - Learns through interaction with environment
        
        Deep Learning, a subset of ML, uses neural networks with multiple layers 
        to model complex patterns in data. Applications include computer vision, 
        natural language processing, and autonomous vehicles.
        
        Ethical Considerations:
        - Bias in training data can lead to unfair outcomes
        - Privacy concerns with data collection
        - Job displacement due to automation
        - Need for transparent and explainable AI systems
        """
        
        # Save processed content
        save_file_content(filename, mock_content)
        
        # Extract mock topics
        mock_topics = [
            "Artificial Intelligence Overview",
            "Machine Learning Fundamentals",
            "Deep Learning Applications",
            "Ethical Considerations in AI",
            "Supervised vs Unsupervised Learning"
        ]
        
        # Save topics
        save_topics(filename, mock_topics)
        
        # Create mock summary
        mock_summary = "This document covers the fundamentals of Artificial Intelligence and Machine Learning, including key concepts, applications, and ethical considerations."
        
        # Save summary
        save_summary(filename, mock_summary)
        
        return {
            "status": "success",
            "message": "PDF processed successfully",
            "topics_count": len(mock_topics),
            "content_length": len(mock_content)
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to process PDF: {str(e)}")

@app.post("/api/embeddings")
async def create_embeddings(request: EmbeddingRequest):
    """Create vector embeddings for document content"""
    try:
        filename = request.filename
        
        # Check if file has been processed
        if filename not in processed_content:
            raise HTTPException(status_code=400, detail="File not processed yet. Please process the file first.")
        
        # Simulate embedding creation
        # In a real implementation, you would use FAISS and HuggingFace embeddings
        content = get_file_content(filename)
        
        # Store mock vector store reference
        vector_stores[filename] = {
            "chunks": len(content.split("\n\n")),  # Mock chunking
            "dimensions": 768,  # Typical embedding dimension
            "created_at": "2026-01-12T10:00:00Z"
        }
        
        return {
            "status": "success",
            "message": "Embeddings created successfully",
            "chunks": vector_stores[filename]["chunks"]
        }
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to create embeddings: {str(e)}")

@app.get("/api/topics")
async def get_topics():
    """Retrieve extracted topics from document"""
    try:
        # For demo, return topics from the last processed document
        # In a real implementation, you would identify the current session/document
        if not topics_data:
            return {"topics": []}
        
        # Return topics from the most recent document
        latest_filename = list(topics_data.keys())[-1]
        topics = get_topics(latest_filename)
        
        return {"topics": topics}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve topics: {str(e)}")

@app.get("/api/summary")
async def get_summary():
    """Retrieve document summary"""
    try:
        # For demo, return summary from the last processed document
        if not summary_data:
            return {"summary": "No document processed yet."}
        
        # Return summary from the most recent document
        latest_filename = list(summary_data.keys())[-1]
        summary = get_summary(latest_filename)
        
        return {"summary": summary}
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Failed to retrieve summary: {str(e)}")

# Question generation functions
async def generate_mcq_questions(topic: str):
    """Generate Multiple Choice Questions"""
    questions = [
        {
            "type": "mcq",
            "data": {
                "question": "What is the primary difference between supervised and unsupervised learning?",
                "option_a": "Supervised learning uses labeled data while unsupervised learning does not",
                "option_b": "Supervised learning is faster than unsupervised learning",
                "option_c": "Unsupervised learning requires more computational power",
                "option_d": "Supervised learning only works with numerical data"
            }
        },
        {
            "type": "mcq",
            "data": {
                "question": "Which of the following is NOT a type of machine learning?",
                "option_a": "Supervised Learning",
                "option_b": "Unsupervised Learning",
                "option_c": "Reinforcement Learning",
                "option_d": "Deterministic Programming"
            }
        }
    ]
    
    for question in questions:
        yield json.dumps(question) + "\n"
        await asyncio.sleep(0.5)  # Simulate processing time

async def generate_true_false_questions(topic: str):
    """Generate True/False Questions"""
    questions = [
        {
            "type": "truefalse",
            "data": {
                "question": "Deep Learning is a subset of Machine Learning."
            }
        },
        {
            "type": "truefalse",
            "data": {
                "question": "Reinforcement Learning is primarily used for clustering unlabeled data."
            }
        }
    ]
    
    for question in questions:
        yield json.dumps(question) + "\n"
        await asyncio.sleep(0.5)

async def generate_short_answer_questions(topic: str):
    """Generate Short Answer Questions"""
    questions = [
        {
            "type": "shortanswer",
            "data": {
                "question": "Explain the concept of overfitting in machine learning."
            }
        },
        {
            "type": "shortanswer",
            "data": {
                "question": "What are the main ethical concerns associated with AI development?"
            }
        }
    ]
    
    for question in questions:
        yield json.dumps(question) + "\n"
        await asyncio.sleep(0.5)

async def generate_problem_solving_questions(topic: str):
    """Generate Problem Solving Questions"""
    questions = [
        {
            "type": "problemsolving",
            "data": {
                "question": "Design a machine learning pipeline for classifying customer reviews as positive or negative. Include data preprocessing steps, model selection criteria, and evaluation metrics."
            }
        }
    ]
    
    for question in questions:
        yield json.dumps(question) + "\n"
        await asyncio.sleep(0.5)

async def generate_essay_questions(topic: str):
    """Generate Essay Questions"""
    questions = [
        {
            "type": "essay",
            "data": {
                "question": "Discuss the impact of artificial intelligence on employment across different sectors. Include both positive and negative effects, and propose strategies for managing this transition."
            }
        }
    ]
    
    for question in questions:
        yield json.dumps(question) + "\n"
        await asyncio.sleep(0.5)

@app.get("/api/questions")
async def generate_questions(topic: str, types: str):
    """Generate questions based on topic and types"""
    try:
        question_types = types.split(',')
        
        async def question_generator():
            # Map question types to generator functions
            generators = {
                'mcq': generate_mcq_questions,
                'true_false': generate_true_false_questions,
                'short_answer': generate_short_answer_questions,
                'problem_solving': generate_problem_solving_questions,
                'essay': generate_essay_questions
            }
            
            # Generate questions for each requested type
            for qtype in question_types:
                if qtype in generators:
                    async for question in generators[qtype](topic):
                        yield question
        
        return StreamingResponse(question_generator(), media_type="application/json")
    except Exception as e:
        error_response = {
            "type": "error",
            "data": {
                "message": f"Failed to generate questions: {str(e)}"
            }
        }
        return StreamingResponse(iter([json.dumps(error_response) + "\n"]), media_type="application/json")

# Answer evaluation functions
async def evaluate_mcq_answer(question_data: dict, user_answer: str):
    """Evaluate Multiple Choice Question"""
    # In a real implementation, you would compare with correct answer
    score = 100 if user_answer.upper() in ['A', 'B'] else 50  # Mock scoring
    reasoning = "Your answer shows good understanding of the concept." if score > 70 else "Consider reviewing the fundamental differences between learning types."
    suggestions = "Review the core concepts of supervised vs unsupervised learning." if score <= 70 else "Continue exploring advanced ML concepts."
    
    return {
        "score": score,
        "reasoning": reasoning,
        "suggestions": suggestions
    }

async def evaluate_true_false_answer(question_data: dict, user_answer: str):
    """Evaluate True/False Question"""
    # In a real implementation, you would compare with correct answer
    score = 100 if user_answer.lower() == 'true' else 50  # Mock scoring
    reasoning = "Correct! Deep Learning is indeed a subset of Machine Learning." if score > 70 else "Review the relationship between different ML approaches."
    suggestions = "Study the hierarchy of AI, ML, and Deep Learning concepts." if score <= 70 else "Explore advanced deep learning architectures."
    
    return {
        "score": score,
        "reasoning": reasoning,
        "suggestions": suggestions
    }

async def evaluate_short_answer(question_data: dict, user_answer: str):
    """Evaluate Short Answer Question"""
    # In a real implementation, you would use LLM to evaluate
    # Mock scoring based on answer length and keywords
    keywords = ['overfitting', 'model', 'training', 'validation'] if 'overfitting' in question_data['question'] else ['ethics', 'bias', 'privacy', 'fairness']
    keyword_matches = sum(1 for kw in keywords if kw.lower() in user_answer.lower())
    score = min(50 + (keyword_matches * 15), 100)
    
    reasoning = f"Your answer covers {keyword_matches} key concepts." if keyword_matches > 0 else "Your answer needs more specific details."
    suggestions = f"Include more terms like: {', '.join(keywords)}" if keyword_matches < len(keywords) else "Excellent comprehensive answer!"
    
    return {
        "score": score,
        "reasoning": reasoning,
        "suggestions": suggestions
    }

async def evaluate_problem_solving(question_data: dict, user_answer: str):
    """Evaluate Problem Solving Question"""
    # In a real implementation, you would use LLM to evaluate
    # Mock scoring based on answer structure
    has_steps = 'step' in user_answer.lower() or '1.' in user_answer or 'first' in user_answer.lower()
    has_methodology = any(term in user_answer.lower() for term in ['preprocess', 'model', 'evaluate', 'metric'])
    score = 70 + (20 if has_steps else 0) + (10 if has_methodology else 0)
    
    reasoning = "Good structured approach to problem solving." if has_steps else "Consider organizing your solution in clear steps."
    suggestions = "Structure your solution with clear steps: problem definition, approach, implementation, evaluation." if not has_steps else "Your methodology is well-structured."
    
    return {
        "score": score,
        "reasoning": reasoning,
        "suggestions": suggestions
    }

async def evaluate_essay(question_data: dict, user_answer: str):
    """Evaluate Essay Question"""
    # In a real implementation, you would use LLM to evaluate
    # Mock scoring based on answer length and content
    word_count = len(user_answer.split())
    has_positive = 'positive' in user_answer.lower() or 'benefit' in user_answer.lower()
    has_negative = 'negative' in user_answer.lower() or 'concern' in user_answer.lower()
    
    score = min(30 + (word_count // 10), 100) if word_count > 50 else 40
    if has_positive and has_negative:
        score = min(score + 20, 100)
    
    reasoning = f"Your essay has good length ({word_count} words) and covers both aspects." if has_positive and has_negative else "Consider addressing both positive and negative impacts."
    suggestions = "Ensure your essay covers both positive and negative impacts with specific examples." if not (has_positive and has_negative) else "Well-balanced comprehensive essay."
    
    return {
        "score": score,
        "reasoning": reasoning,
        "suggestions": suggestions
    }

@app.post("/api/evaluate")
async def evaluate_answers(request: EvaluationRequest):
    """Evaluate user answers"""
    try:
        # Mock question data for evaluation
        mock_questions = {
            0: {"type": "mcq", "data": {"question": "What is the primary difference between supervised and unsupervised learning?"}},
            1: {"type": "truefalse", "data": {"question": "Deep Learning is a subset of Machine Learning."}},
            2: {"type": "shortanswer", "data": {"question": "Explain the concept of overfitting in machine learning."}},
            3: {"type": "problemsolving", "data": {"question": "Design a machine learning pipeline for classifying customer reviews as positive or negative."}},
            4: {"type": "essay", "data": {"question": "Discuss the impact of artificial intelligence on employment across different sectors."}}
        }
        
        async def evaluation_generator():
            # Map question types to evaluation functions
            evaluators = {
                'mcq': evaluate_mcq_answer,
                'truefalse': evaluate_true_false_answer,
                'shortanswer': evaluate_short_answer,
                'problemsolving': evaluate_problem_solving,
                'essay': evaluate_essay
            }
            
            # Evaluate each answer
            for answer_submission in request.answers:
                question_index = answer_submission.questionIndex
                user_answer = answer_submission.answer
                
                # Get question data
                if question_index in mock_questions:
                    question_data = mock_questions[question_index]
                    question_type = question_data["type"]
                    
                    # Evaluate answer
                    if question_type in evaluators:
                        evaluation = await evaluators[question_type](question_data["data"], user_answer)
                        yield json.dumps(evaluation) + "\n"
                        await asyncio.sleep(0.3)  # Simulate processing time
                else:
                    # Error evaluation
                    error_evaluation = {
                        "score": 0,
                        "reasoning": "Question not found for evaluation.",
                        "suggestions": "Ensure you're answering the correct questions."
                    }
                    yield json.dumps(error_evaluation) + "\n"
                    await asyncio.sleep(0.3)
        
        return StreamingResponse(evaluation_generator(), media_type="application/json")
    except Exception as e:
        error_response = {
            "score": 0,
            "reasoning": f"Failed to evaluate answers: {str(e)}",
            "suggestions": "Please try submitting your answers again."
        }
        return StreamingResponse(iter([json.dumps(error_response) + "\n"]), media_type="application/json")

@app.post("/api/ask")
async def ask_question(request: QuestionRequest):
    """Answer questions about the document"""
    try:
        question = request.question.lower()
        
        # Mock responses based on question content
        async def answer_generator():
            if "difference" in question and ("supervised" in question or "unsupervised" in question):
                response = "Supervised learning uses labeled training data where the correct answers are provided during training. The model learns to map inputs to outputs based on these examples. Unsupervised learning works with unlabeled data, where the algorithm must find patterns and structure in the data without knowing the correct answers.\n\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)  # Simulate typing
                
                response = "Key differences:\n1. Supervised learning has a target variable, unsupervised does not\n2. Supervised learning is used for prediction, unsupervised for discovery\n3. Supervised learning examples: classification, regression\n4. Unsupervised learning examples: clustering, association rules\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
            
            elif "deep learning" in question:
                response = "Deep learning is a subset of machine learning that uses neural networks with multiple layers (hence 'deep'). These networks can automatically learn hierarchical representations of data.\n\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
                
                response = "Key characteristics:\n1. Uses artificial neural networks\n2. Can process complex data like images, audio, text\n3. Requires large amounts of data\n4. Automatically extracts features\n5. Examples: CNNs for images, RNNs for sequences\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
            
            elif "ethics" in question or "ethical" in question:
                response = "Ethical considerations in AI include:\n\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
                
                response = "1. Bias and Fairness: Ensuring AI systems don't discriminate\n2. Privacy: Protecting personal data used in training\n3. Transparency: Making AI decisions explainable\n4. Accountability: Determining responsibility for AI actions\n5. Job Displacement: Managing impact on employment\n6. Safety: Ensuring AI systems behave as intended\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
            
            else:
                response = "I understand your question about: " + request.question + "\n\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
                
                response = "Based on the document content, this topic relates to artificial intelligence and machine learning concepts. For a more specific answer, please ask about particular aspects like supervised learning, deep learning architectures, or ethical considerations in AI development.\n\n"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
                
                response = "Would you like me to elaborate on any specific aspect of AI or machine learning?"
                for word in response.split():
                    yield word + " "
                    await asyncio.sleep(0.05)
        
        return StreamingResponse(answer_generator(), media_type="text/plain")
    except Exception as e:
        error_response = f"Error processing your question: {str(e)}\nPlease try rephrasing your question."
        return StreamingResponse(iter([error_response]), media_type="text/plain")

# Health check endpoint
@app.get("/health")
async def health_check():
    return {"status": "healthy", "message": "QuestionCraft API is running"}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)