# GitHub Repository Analyzer

A comprehensive AI-powered tool for analyzing GitHub repositories using advanced NLP techniques, embeddings, and interactive chat capabilities. This project enables users to explore, understand, and interact with codebases through an intuitive Streamlit interface and powerful backend processing.

## Features Mapped with Development Conversations

### **Conversation 1: Foundation & Code Processing**
- **GitHub Repository Fetching:** Implemented repository cloning with size validation and error handling
- **Multi-Language Code Parsing:** Added support for Python, React, JavaScript, and Docker files
- **Intelligent Chunking System:** Created function and class-level code segmentation for better analysis
- **Binary File Detection:** Fixed issues with binary file handling and filtering
- **Main Orchestrator:** Built central entry point for coordinating all analysis components

**Screenshots:**
- [Creating chunks and displaying stats](https://drive.google.com/file/d/1SE4UQ8XzGSjI85pkdnsnIRlJNaQHbETY/view?usp=share_link)
- [Main Orchestrator](https://drive.google.com/file/d/1hpjFSfeniKpLNKwNwH502jYNqal_30Zg/view?usp=share_link)
- [Unit Tests](https://drive.google.com/file/d/1GmtbEZkpLMZLYqO-HIbleG13trMK5UFg/view?usp=share_link)

### **Conversation 2: Embeddings & Vector Storage**
- **OpenAI Integration:** Implemented text-embedding-3-small model for generating semantic embeddings
- **Chunk Processing Pipeline:** Added metadata integration and token limit handling
- **In-Memory Caching:** Optimized embedding generation with intelligent caching mechanisms
- **ChromaDB Integration:** Built persistent vector storage with repository-based collections
- **Token Management:** Fixed token limit issues with smart chunk splitting

**Screenshots:**
- [Embeddings Generation](https://drive.google.com/file/d/1FQ2SLvHKHH9SorT7DN6QajCXthgceagk/view?usp=share_link)
- [Caching](https://drive.google.com/file/d/1NCB-nC1oxYg3zD6EOQsEpYDEu4SWS8H2/view?usp=share_link)
- [Vector Storage](https://drive.google.com/file/d/17y_T8_pQt3ld7evyVpfF4UtuChGxEYrT/view?usp=share_link)
- [Unit Tests](https://drive.google.com/file/d/17_KIRixf3Ix3i9_pVIpgSJcE2qnCx5Ja/view?usp=share_link)

### **Conversation 3: NLP Chat System**
- **Interactive Query System:** Built two-pass query processing for accurate code analysis
- **Citation & Referencing:** Implemented source attribution for all responses
- **Context-Aware Responses:** Added repository-specific context understanding
- **Memory Management:** Fixed integration issues and optimized resource usage
- **Command-Line Interface:** Created interactive CLI for repository querying

**Screenshots:**
- [List Repos](https://drive.google.com/file/d/1OSpPvqpukeigGytn1dZyAf2sib8nxtfj/view?usp=share_link)
- [Query](https://drive.google.com/file/d/1Syv5IPgtckKsqmD-YlsZAosKPPDZ1Q6E/view?usp=share_link)
- [Interactive mode](https://drive.google.com/file/d/1GRkygX8NLVUDU_PjbZIsW6oHOZJCDJSr/view?usp=share_link)
- [Unit Tests](https://drive.google.com/file/d/17J9eLftczUBR3nMlrCfNLn_Gtos-rmHM/view?usp=share_link)

### **Conversation 4: Streamlit Web Interface**
- **Modern Web UI:** Implemented responsive Streamlit interface for repository analysis
- **Repository Input System:** Added URL input with validation and progress tracking
- **Repository Listing:** Created overview page showing all analyzed repositories
- **Integrated Chat Interface:** Built seamless chat experience using existing query functions
- **Architecture Consistency:** Fixed import paths and maintained clean code structure

**Screenshots:**
- [Home Page](https://drive.google.com/file/d/1MypbF6CbuSv9A_L3Zz0zwYsrzu-rMjA8/view?usp=share_link)
- [Fetching Repository from GitHub](https://drive.google.com/file/d/1l6KJLNIOtRWCw6B-yFldHD2HiV0-cqFg/view?usp=share_link)
- [List repos](https://drive.google.com/file/d/196o3S3wiJvvPtjQ08wHTHEjXW2RkgN8g/view?usp=share_link)
- [Chat](https://drive.google.com/file/d/1hbURid67OLTMoqHLml_Y6Li8VF5Vnyjv/view?usp=share_link)
- [Unit Tests](https://drive.google.com/file/d/1V2gFKuDETXxezSHbUpbr82CQBe1UBIoJ/view?usp=share_link)

### **Conversation 5: Advanced Code Parsing**
- **Tree-sitter Integration:** Implemented language-agnostic parsing for improved accuracy
- **Parallel Processing:** Added thread-safe mechanisms for faster analysis
- **File Path Management:** Optimized file handling and import structures
- **Multi-Language Support:** Enhanced parsing pipeline for large, diverse repositories
- **Performance Optimization:** Improved processing speed for complex codebases

### **Conversation 6: GitHub OAuth & Private Repositories**
- **OAuth Integration:** Implemented secure GitHub authentication flow
- **Private Repository Access:** Added support for analyzing private repositories
- **Repository Listing:** Built interface showing user's accessible repositories
- **Authentication State Management:** Fixed state parameter handling and session management
- **Seamless Integration:** Connected OAuth flow with existing analysis pipeline

**Screenshots:**
- [Connect GitHub](https://drive.google.com/file/d/1EIO6hg-_tysJoV6yce6NPHh9HuMSggpV/view?usp=share_link)
- [GitHub Repository List](https://drive.google.com/file/d/198uaeSHGigX0BzwB5_rjBExvbaQlpFUo/view?usp=share_link)
- [Analyze GitHub Repository](https://drive.google.com/file/d/1v3ugNAhnqeHD5gmRAP1HVE17alaRUXnN/view?usp=share_link)

### **Conversation 7: Advanced Repository Exploration**
- **File Tree Explorer:** Implemented lazy-loading directory browser
- **On-Demand Summaries:** Added LLM-powered file content summarization
- **Advanced Search:** Built file and content search functionality across repositories
- **File-Scoped Chat:** Created targeted chat interactions for specific files
- **Error Handling:** Added robust retry mechanisms and user feedback
- **UI Improvements:** Fixed alignment issues and enhanced user experience

**Screenshots:**
- [List Repo](https://drive.google.com/file/d/1OkSYD8PHGM3YmTupce0F0IRB8BV4liuC/view?usp=share_link)
- [Repo Explore](https://drive.google.com/file/d/1bLyvfNZIkldYjWaAFLcuxJy47k_S98oZ/view?usp=share_link)
- [Search in Repo](https://drive.google.com/file/d/1aamSmk2DoJ8dvlvpEdURZ1N6giJAjslO/view?usp=share_link)
- [File Chat](https://drive.google.com/file/d/1NxnwbBbXPPaZ8-cf02QRqv4rXS_QDXBa/view?usp=share_link)
- [File summary](https://drive.google.com/file/d/1xGx_ZLoZKUhvtqEM14yeIBh_0SZGGoeD/view?usp=share_link)
- [Unit Tests](https://drive.google.com/file/d/15kEGhtLnsPZi77vpFnSx71vkuIFdIcit/view?usp=share_link)

### **Conversation 8: Production Deployment**
- **Docker Containerization:** Implemented full containerization with Docker and Docker Compose
- **Permission Management:** Fixed directory permissions for data persistence
- **CI/CD Pipeline:** Set up automated deployment with GitHub Actions
- **SSL Configuration:** Configured secure HTTPS with Certbot and DNS verification
- **Scalable Architecture:** Designed multi-user support and horizontal scaling

## Project Structure

```
.
├── .dockerignore         # Docker ignore file
├── .env                  # Environment variables
├── .gitignore            # Git ignore file
├── Dockerfile            # Docker configuration
├── README.md             # Project documentation
├── data/                 # Directory for storing data files
├── docker-compose.yml    # Docker compose configuration
├── requirements-test.txt # Test dependencies
├── requirements.txt      # Project dependencies
├── setup.py              # Package configuration
├── src/                  # Source code
│   ├── auth/             # Authentication modules
│   ├── embeddings/       # Embedding generation and management
│   ├── fetcher/          # Code for fetching repositories
│   ├── llm/              # Language model integration
│   ├── parser/           # Code parsing and analysis
│   ├── processing/       # Data processing pipelines
│   ├── query/            # Query handling and processing
│   ├── ui/               # User interface components
│   ├── utils/            # Utility functions
│   └── main.py           # Main application entry point
├── streamlit_app.py      # Streamlit application entry point
└── tests/                # Test files
    └── ui/               # UI test files
```

## Prerequisites

- **Python 3.8+** with pip package manager
- **Docker & Docker Compose** for containerized deployment
- **Git** for repository cloning
- **OpenAI API Key** for embedding generation and chat functionality
- **GitHub OAuth App** (for private repository access)

## Installation & Setup

### 1. Clone the Repository
```bash
git clone https://github.com/your-username/github-repo-analyzer.git
cd github-repo-analyzer
```

### 2. Environment Configuration
```bash
# Copy environment template
cp .env.example .env

# Edit with your configuration
nano .env
```

Required environment variables:
```env
OPENAI_API_KEY=your_openai_api_key_here
GITHUB_CLIENT_ID=your_github_oauth_client_id
GITHUB_CLIENT_SECRET=your_github_oauth_client_secret
CHROMA_PERSIST_DIRECTORY=./data/embeddings
REPOS_DIRECTORY=./data/repos
```

### 3. Installation Options

#### Local Development
```bash
# Install Python dependencies
pip install -r requirements.txt

# Run the Streamlit application
streamlit run src/ui/streamlit_app.py
```

## Technologies Used

- **Python 3.8+** - Primary programming language
- **Streamlit** - Web interface framework
- **OpenAI API** - Embeddings and chat completion
- **ChromaDB** - Vector database for embeddings
- **Tree-sitter** - Language-agnostic code parsing

### Integration & Deployment
- **GitHub API** - Repository access and OAuth
- **Docker & Docker Compose** - Containerization
- **GitHub Actions** - CI/CD pipeline
- **Certbot** - SSL certificate management

## Final Project Demo
[Final Project Demo](https://drive.google.com/file/d/1Ztl7zsn7i8HKYB0zCvWr39LfZ6F-zJlG/view?usp=share_link)