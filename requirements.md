# Agent Dependencies

This document lists all the required Python packages for each agent in the project. You can install these packages using either pip or conda.

## Common Dependencies
These packages are required by all agents:
```
python-dotenv>=0.19.0
langchain>=0.0.267
openai>=0.27.0
```

## Code Agent Dependencies
Required for code analysis and documentation:
```
ast>=0.0.0  # Built-in
pathlib>=1.0.1  # Built-in
```

## Data Agent Dependencies
Required for data processing and analysis:
```
pandas>=1.5.0
numpy>=1.21.0
matplotlib>=3.5.0
seaborn>=0.11.0
openpyxl>=3.0.0  # For Excel file support
```

## Document Agent Dependencies
Required for document processing and analysis:
```
PyPDF2>=3.0.0
faiss-cpu>=1.7.0  # For vector storage
```

## Task Agent Dependencies
Required for task automation and scheduling:
```
schedule>=1.1.0
requests>=2.26.0
```

## Web Agent Dependencies
Required for web research and content extraction:
```
beautifulsoup4>=4.9.0
requests>=2.26.0
faiss-cpu>=1.7.0  # For vector storage
google-search-results>=2.4.0  # For Serper API
```

## Installation Instructions

### Using pip
```bash
# Install all dependencies
pip install -r requirements.txt

# Or install specific agent dependencies
pip install python-dotenv langchain openai  # Common dependencies
pip install pandas numpy matplotlib seaborn openpyxl  # Data agent
pip install PyPDF2 faiss-cpu  # Document agent
pip install schedule requests  # Task agent
pip install beautifulsoup4 requests faiss-cpu google-search-results  # Web agent
```

### Using conda
```bash
# Create a new environment
conda create -n agentic_ai python=3.9

# Activate the environment
conda activate agentic_ai

# Install packages
conda install -c conda-forge python-dotenv langchain openai
conda install -c conda-forge pandas numpy matplotlib seaborn openpyxl
conda install -c conda-forge pypdf2 faiss-cpu
conda install -c conda-forge schedule requests
conda install -c conda-forge beautifulsoup4 requests faiss-cpu
pip install google-search-results  # Not available in conda
```

## Environment Variables
The following environment variables need to be set:
- `OPENAI_API_KEY`: Your OpenAI API key
- `SERPER_API_KEY`: Your Serper API key (for web agent)
- `EMAIL_USER`: Email address for notifications (for task agent)
- `EMAIL_PASSWORD`: Email password for notifications (for task agent)
- `SMTP_SERVER`: SMTP server address (default: smtp.gmail.com)
- `SMTP_PORT`: SMTP server port (default: 587)

Create a `.env` file in the project root with these variables:
```
OPENAI_API_KEY=your_openai_api_key
SERPER_API_KEY=your_serper_api_key
EMAIL_USER=your_email@example.com
EMAIL_PASSWORD=your_email_password
SMTP_SERVER=smtp.gmail.com
SMTP_PORT=587
``` 