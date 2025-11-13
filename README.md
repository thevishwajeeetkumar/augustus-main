# Augustus - AI-Powered YouTube Learning Agent

> Augustus is an AI agent for learning through YouTube videos. On the surface it looks simple, but it has agentic capabilities where it decides on searching the web at any step of answering your queries. It has contextual awareness and a world-class RAG system backed by Pinecone.

[![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)](https://www.python.org)
[![FastAPI](https://img.shields.io/badge/FastAPI-0.110+-green.svg)](https://fastapi.tiangolo.com)
[![License](https://img.shields.io/badge/license-MIT-blue.svg)](LICENSE)

---

## ✨ Features

### **🤖 Intelligent AI Agent**
- **Context-Aware Retrieval**: Agent orchestrates YouTube RAG and augments answers with Google Custom Search when necessary
- **Conversation Memory**: Maintains conversation history across sessions
- **Multi-Language Support**: Handles Hindi and English video transcripts

### **📚 World-Class RAG System**
- **Pinecone Vector Database**: User-specific isolated indexes
- **Advanced Retrieval**: Multi-query retrieval with contextual compression
- **Semantic Search**: Powered by OpenAI embeddings (text-embedding-3-large)

### **🔐 Enterprise-Grade Authentication**
- **JWT Tokens**: Secure OAuth2 implementation with scopes
- **User Management**: Registration, authentication, role-based access
- **Session Management**: Persistent conversation contexts

### **🗄️ Cloud-Native Architecture**
- **NeonDB**: Serverless PostgreSQL with auto-suspend
- **Alembic Migrations**: Version-controlled database schema
- **Production-Ready**: Optimized connection pooling, SSL/TLS

---

## 🚀 Quick Start

### **Prerequisites**

- Python 3.10+
- OpenAI API key
- Pinecone API key
- NeonDB account (or use local PostgreSQL)

### **Installation**

```bash
# Clone the repository
git clone https://github.com/piyush2118/augustus.git
cd augustus

# Create virtual environment
python -m venv myvenv
source myvenv/bin/activate  # On Windows: myvenv\Scripts\activate

# Install dependencies
pip install -r requirements.txt
```

### **Configuration**

Create a `.env` file:

```bash
# Copy example file
cp env.example .env

# Edit with your credentials
nano .env
```

Required environment variables:
```env
# Database (NeonDB or local PostgreSQL)
DATABASE_URL=postgresql://user:password@host:5432/database?sslmode=require

# Security
SECRET_KEY=your-64-char-random-key  # Generate: openssl rand -hex 32

# API Keys
OPENAI_API_KEY=sk-your-openai-key
PINECONE_API_KEY=your-pinecone-key

# Environment
ENVIRONMENT=production
ACCESS_TOKEN_EXPIRE_MINUTES=30
```

### **Database Setup**

```bash
# Run migrations to create tables
alembic upgrade head
```

### **Run the Application**

```bash
# Start the server
uvicorn main:app --reload

# Visit API documentation
open http://localhost:8000/docs
```

---

## 📖 API Documentation

### **Authentication**

#### **Sign Up**
```http
POST /signup
Content-Type: application/json

{
  "username": "johndoe",
  "email": "john@example.com",
  "password": "SecurePassword123!"
}
```

#### **Sign In (Get JWT Token)**
```http
POST /token
Content-Type: application/x-www-form-urlencoded

username=johndoe&password=SecurePassword123!
```

**Response**:
```json
{
  "access_token": "eyJ0eXAiOiJKV1...",
  "token_type": "bearer",
  "expires_in": 1800
}
```

### **YouTube Q&A**

```http
POST /
Authorization: Bearer YOUR_JWT_TOKEN
Content-Type: application/json

{
  "query": "What is this video about?",
  "url": "https://www.youtube.com/watch?v=dQw4w9WgXcQ",
  "session_id": "optional-session-id-for-continuity"
}
```

**Response**:
```json
{
  "answer": "Based on the video transcript, this video discusses...",
  "session_id": "uuid-of-conversation-session",
  "video_context": "video-id"
}
```

### **Health Check**

```http
GET /health
```

---

## 🏗️ Architecture

### **System Components**

```
┌─────────────────┐         ┌──────────────────┐         ┌─────────────────┐
│   Client API    │────────>│   FastAPI App    │────────>│   NeonDB        │
│  (JWT Auth)     │         │   (Augustus)     │         │  (PostgreSQL)   │
└─────────────────┘         └────────┬─────────┘         └─────────────────┘
                                     │
                    ┌────────────────┼────────────────┐
                    │                │                │
               ┌────▼─────┐    ┌────▼─────┐    ┌────▼──────┐
               │ OpenAI   │    │ Pinecone │    │ Google    │
               │(GPT-4o)  │    │ (Vector  │    │ Custom    │
               │          │    │  Store)  │    │  Search   │
               └──────────┘    └──────────┘    └───────────┘
```

### **Data Flow**

1. **User Authentication**: JWT token validation
2. **Session Management**: Retrieve or create user session
3. **Video Processing**: Extract transcript, translate if needed
4. **Embedding**: Store in user-specific Pinecone index
5. **Agent Reasoning**: Retrieve and synthesize context from the YouTube knowledge base and Google Custom Search
6. **Response Generation**: GPT-4o-mini generates answer
7. **Context Storage**: Save conversation in PostgreSQL

---

## 🗄️ Database Schema

### **Users Table**
- Authentication and user management
- UUID primary key
- bcrypt password hashing
- OAuth2 scopes (read, write, admin)

### **User Sessions Table**
- Conversation context tracking
- Video ID tracking for follow-up questions
- 24-hour expiration with auto-cleanup

### **Messages Table**
- Conversation history (sliding window)
- Human/AI/System/Tool messages
- JSONB metadata for flexibility

---

## 🔒 Security Features

- ✅ JWT authentication with OAuth2 scopes
- ✅ bcrypt password hashing
- ✅ Strong SECRET_KEY validation
- ✅ SSL/TLS encryption (NeonDB)
- ✅ CORS protection
- ✅ Input validation (Pydantic)
- ✅ SQL injection protection (SQLAlchemy)
- ✅ User-specific data isolation (Pinecone indexes)

---

## 🚀 Deployment

### **Railway**

```bash
railway init
railway variables set DATABASE_URL="your-neondb-url"
railway variables set SECRET_KEY="$(openssl rand -hex 32)"
railway variables set OPENAI_API_KEY="your-key"
railway variables set PINECONE_API_KEY="your-key"
railway variables set ENVIRONMENT="production"
railway up
```

### **Render**

1. Connect GitHub repository
2. Set environment variables in dashboard
3. Deploy

### **Environment Variables**

Required:
- `DATABASE_URL` - NeonDB connection string
- `SECRET_KEY` - 64-char random key
- `OPENAI_API_KEY` - OpenAI API key
- `PINECONE_API_KEY` - Pinecone API key

Optional:
- `ENVIRONMENT` - production/development (default: development)
- `ACCESS_TOKEN_EXPIRE_MINUTES` - JWT expiration (default: 30)
- `CORS_ORIGINS` - Allowed origins (default: *)

---

## 📚 Documentation

- **[Deployment Guide](DEPLOYMENT_READY_GUIDE.md)** - Complete deployment instructions
- **[NeonDB Setup](README_NEONDB.md)** - Cloud database configuration
- **[Authentication Guide](AUTH_README.md)** - Auth system details
- **[Quick Start](QUICK_START.md)** - Getting started guide
- **[Search Tool Update](README_TOOL_CHANGE.md)** - Summary of the DuckDuckGo removal
- **[API Documentation](http://localhost:8000/docs)** - Interactive Swagger UI

---

## 🛠️ Development

### **Project Structure**

```
augustus/
├── main.py                 # FastAPI application
├── database.py             # Database models and configuration
├── yt_to_id.py            # YouTube ID extraction utility
├── requirements.txt        # Python dependencies
├── alembic/               # Database migrations
│   ├── env.py
│   └── versions/
├── alembic.ini            # Alembic configuration
├── env.example            # Environment variable template
└── docs/                  # Documentation
```

### **Running Tests**

```bash
# Test authentication
python test_auth_system.py

# Test OAuth2 endpoints
python test_oauth2_endpoints.py

# Test deployment readiness
python test_deployment_ready.py
```

### **Database Migrations**

```bash
# Create new migration
alembic revision --autogenerate -m "Description"

# Apply migrations
alembic upgrade head

# Rollback migration
alembic downgrade -1
```

---

## 🤝 Contributing

Contributions are welcome! Please feel free to submit a Pull Request.

1. Fork the repository
2. Create your feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- **FastAPI** - Modern web framework
- **LangChain** - LLM framework
- **Pinecone** - Vector database
- **OpenAI** - Language models
- **NeonDB** - Serverless PostgreSQL
- **Alembic** - Database migrations

---

## 📞 Support

For questions or issues:
- Open an issue on GitHub
- Check the documentation in the `docs/` folder
- Review the comprehensive guides (20+ documentation files)

---

## 🌟 Features Roadmap

- [ ] Email verification for user registration
- [ ] Password reset functionality
- [ ] Rate limiting on auth endpoints
- [ ] Webhook support for real-time notifications
- [ ] Multi-video conversation support
- [ ] Export conversation history
- [ ] Advanced analytics dashboard

---

**Built with ❤️ using FastAPI, LangChain, and NeonDB**

#   a u g u s t u s - m a i n 
 
 