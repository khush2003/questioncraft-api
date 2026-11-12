# QuestionCraft FastAPI Backend

This is the backend service for the QuestionCraft application, built with FastAPI.

## Features

- PDF upload and processing
- Document content analysis and topic extraction
- Vector embedding creation with FAISS
- Question generation for multiple question types
- Answer evaluation and scoring
- Document Q&A chat functionality
- Real-time streaming responses

## Setup

### Prerequisites

- Python 3.8+
- pip

### Installation

1. Navigate to the fastapi directory:
   ```bash
   cd fastapi
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

### Running the Server

For development:
```bash
uvicorn main:app --reload --port 8000
```

For production:
```bash
uvicorn main:app --host 0.0.0.0 --port 8000
```

### Using Docker

Build the image:
```bash
docker build -t questioncraft-backend .
```

Run the container:
```bash
docker run -p 8000:8000 questioncraft-backend
```

## API Endpoints

### Upload and Processing
- `POST /api/upload` - Upload a PDF file
- `POST /api/process` - Process the uploaded PDF
- `POST /api/embeddings` - Create vector embeddings

### Content Retrieval
- `GET /api/topics` - Get extracted topics
- `GET /api/summary` - Get document summary

### Question Generation
- `GET /api/questions` - Generate questions (streaming)

### Evaluation
- `POST /api/evaluate` - Evaluate answers (streaming)

### Chat
- `POST /api/ask` - Ask questions about the document (streaming)

## Development

### Adding New Features

1. Add new endpoints in `main.py`
2. Update the Pydantic models as needed
3. Test with the provided API documentation at `http://localhost:8000/docs`

### API Documentation

Once the server is running, visit `http://localhost:8000/docs` for interactive API documentation.

## Environment Variables

- `OPENROUTER_API_KEY` - For OpenRouter LLM access (in full implementation)
- `HF_KEY` - HuggingFace API token for embeddings (in full implementation)

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

## License

This project is licensed for educational use.