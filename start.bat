@echo off
echo 🤖 Multi-Source RAG Chatbot - Quick Start Script
echo ================================================
echo.

REM Check if .env file exists
if not exist .env (
    echo ⚠️  No .env file found. Creating from .env.example...
    copy .env.example .env
    echo ✅ Created .env file
    echo.
    echo 🔑 IMPORTANT: Edit the .env file and add your OpenAI API key!
    echo    Open .env file in notepad and add your key
    echo.
    pause
)

REM Check if Docker is running
docker info >nul 2>&1
if errorlevel 1 (
    echo ❌ Docker is not running. Please start Docker Desktop first.
    pause
    exit /b 1
)

echo 🐳 Starting Docker containers...
docker-compose up --build -d

echo.
echo ⏳ Waiting for services to be ready...
timeout /t 10 /nobreak > nul

echo.
echo ✅ Application should be running!
echo.
echo 📍 Access the application at:
echo    🌐 Web Interface: http://localhost:8080
echo    📚 API Docs: http://localhost:8080/docs
echo    🔍 Health Check: http://localhost:8080/api/health
echo.
echo 📊 View logs:
echo    docker-compose logs -f app
echo.
echo 🛑 Stop the application:
echo    docker-compose down
echo.
pause
