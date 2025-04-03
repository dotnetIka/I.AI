# AI Question Answering API

A FastAPI application that integrates OpenAI and Qdrant vector database to provide question answering capabilities about Georgian history (1918-1921).

## Features

- Predefined knowledge base about the Democratic Republic of Georgia (1918-1921)
- Answer questions based on stored documents using OpenAI's GPT models
- Semantic search using cosine similarity
- Docker and Docker Compose support for easy deployment

## Prerequisites

### Docker Setup

1. Install Docker Desktop:
   - Download from [Docker Desktop website](https://www.docker.com/products/docker-desktop)
   - Run the installer
   - Start Docker Desktop
   - Wait for Docker to fully initialize (check the whale icon in system tray)

2. Verify Docker installation:
```bash
docker --version
docker-compose --version
```

3. Make sure Docker Desktop is running:
   - Look for the Docker whale icon in your system tray
   - It should show "Docker Desktop is running"
   - If not, right-click the icon and select "Start"

- OpenAI API key

## Security

### Protecting Your API Keys

1. Never commit your `.env` file to version control
2. Copy `.env.example` to `.env` and fill in your actual values:
```bash
cp .env.example .env
```

3. Add the following to your `.env` file:
```bash
OPENAI_API_KEY=your_openai_api_key_here
QDRANT_URL=http://qdrant:6333
QDRANT_COLLECTION_NAME=documents
```

4. Make sure `.env` is in your `.gitignore` file

## Development Setup

1. Clone the repository:
```bash
git clone <repository-url>
cd <repository-name>
```

2. Set up your environment variables as described in the Security section above

3. Install development dependencies:
```bash
pip install -r requirements-dev.txt
```

4. Install pre-commit hooks:
```bash
pre-commit install
```

5. Start the application:
```bash
docker-compose up --build
```

The API will be available at `http://localhost:8000`

## API Endpoints

### POST /generate-embeddings
Generate embeddings for the predefined Georgian history text and store them in Qdrant. This endpoint doesn't require any parameters.

### POST /ask
Ask a question about the Democratic Republic of Georgia (1918-1921) and get an answer based on the stored documents.

Request body:
```json
{
    "question": "Your question here"
}
```

## API Documentation

Once the server is running, you can access the API documentation at:
- Swagger UI: `http://localhost:8000/docs`
- ReDoc: `http://localhost:8000/redoc`

## Docker Services

The application consists of two Docker services:

1. `api`: The FastAPI application
   - Exposed on port 8000
   - Automatically restarts unless stopped
   - Mounts the current directory for development

2. `qdrant`: The vector database
   - Exposed on ports 6333 and 6334
   - Persists data in a Docker volume
   - Automatically restarts unless stopped

## Troubleshooting

### Docker Issues

1. If you see "Docker Desktop is not running":
   - Open Docker Desktop
   - Wait for it to fully initialize
   - Check the system tray icon status

2. If you see connection errors:
   - Make sure Docker Desktop is running
   - Try restarting Docker Desktop
   - Check if your antivirus is blocking Docker

3. If containers won't start:
   - Check if ports 8000, 6333, and 6334 are available
   - Try stopping other Docker containers
   - Check Docker logs for errors

## Notes

- The application and Qdrant are configured to restart automatically unless explicitly stopped
- Qdrant data is persisted in a Docker volume named `qdrant_data`
- The API service mounts the current directory as a volume for development purposes
- Environment variables are passed from the host's `.env` file to the containers 