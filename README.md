# DMP-AIHelpdeskSystem
Base of the RAG-GrahpRAG system created as part of the DMP @ SPSE Plzen. 

## Repository structure
```plaintext
.
├── .git/                     # Git version control directory
├── rag-system/               # Main directory for the RAG system
│   ├── .idea/                # IDE-specific files
│   ├── chat_app/             # Flutter-based chat application
|   |   ├── android/              # Android platform-specific files
|   |   ├── build/                # Build files
|   |   ├── ios/                  # iOS platform-specific files
|   |   ├── lib/                  # Main Dart source code for the app
|   |   ├── linux/                # Linux platform files
|   |   ├── macos/                # macOS platform files
|   |   ├── web/                  # Web platform files
|   |   ├── windows/              # Windows platform files
|   |   ├── analysis_options.yaml # Dart analysis configuration
|   |   ├── pubspec.lock          # Dart package lock file
|   |   ├── pubspec.yaml          # Dart package configuration
|   |   └── README.md             # Documentation for the chat app
│   ├── rag-grag/             # Graph-based RAG system implementation
|   │   ├── __pycache__/          # Python cache files
│   │   ├── postgres-data/        # PostgreSQL data files
│   │   ├── tests/                # Test scripts for backend components
│   │   ├── volumes/              # Docker volumes for persistence
│   │   ├── backend.py            # Backend service script
│   │   ├── docker-compose.yml    # Docker Compose configuration
│   │   ├── frontend.py           # Frontend service script
│   │   └── init.sql              # SQL initialization script
│   ├── scraper/              # Data scraper and document processing scripts
│   └── venv/                 # Python virtual environment
├── zadani/                   # Additional task-related files
├── .gitignore                # Git ignore rules
└── README.md                 # Project documentation
```
## Installation
### RAG-GRAG Backend
1. Use `docker-compose up` in `rag-system/rag-grag/` directory
2. Install python dependecies using `pip install -r requirements.txt`in `rag-system/rag-grag/` directory

---
By [Marek Ruttner](https://www.linkedin.com/in/marek-ruttner/) 2024