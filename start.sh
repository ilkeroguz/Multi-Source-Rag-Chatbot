#!/bin/bash

echo "🤖 Multi-Source RAG Chatbot - Quick Start Script"
echo "================================================"
echo ""

# Check if .env file exists
if [ ! -f .env ]; then
    echo "⚠️  No .env file found. Creating from .env.example..."
    cp .env.example .env
    echo "✅ Created .env file"
    echo ""
    echo "🔑 IMPORTANT: Edit the .env file and add your OpenAI API key!"
    echo "   Run: nano .env (or use any text editor)"
    echo ""
    read -p "Press Enter after you've added your API key to continue..."
fi

# Check if Docker is installed
if ! command -v docker &> /dev/null; then
    echo "❌ Docker is not installed. Please install Docker first."
    echo "   Visit: https://docs.docker.com/get-docker/"
    exit 1
fi

# Check if docker-compose is installed
if ! command -v docker-compose &> /dev/null; then
    echo "❌ docker-compose is not installed. Please install docker-compose first."
    echo "   Visit: https://docs.docker.com/compose/install/"
    exit 1
fi

echo "🐳 Starting Docker containers..."
docker-compose up --build -d

echo ""
echo "⏳ Waiting for services to be ready..."
sleep 10

# Check if services are running
if docker ps | grep -q "rag-chatbot-app"; then
    echo ""
    echo "✅ Application is running!"
    echo ""
    echo "📍 Access the application at:"
    echo "   🌐 Web Interface: http://localhost:8080"
    echo "   📚 API Docs: http://localhost:8080/docs"
    echo "   🔍 Health Check: http://localhost:8080/api/health"
    echo ""
    echo "📊 View logs:"
    echo "   docker-compose logs -f app"
    echo ""
    echo "🛑 Stop the application:"
    echo "   docker-compose down"
    echo ""
else
    echo "❌ Failed to start application. Check logs:"
    echo "   docker-compose logs"
fi
