
# Overview
DocMind AI is an intelligent document analysis and processing system that leverages advanced NLP and machine learning techniques to extract insights, automate workflows, and enhance document management capabilities.


## Features

### Document Processing

- Multi-format Support: Process PDF, DOCX, TXT, CSV, and Excel files
- OCR Integration: Extract text from images and scanned documents
- Table Recognition: Automatically extract and structure tabular data

### AI-Powered Analysis

- Natural Language Understanding: Respond to queries about document content
- Key Information Extraction: Identify and highlight critical information
- Summarization: Generate concise document summaries

### Data Visualization

Interactive Charts: Create visualizations from document data

- Line plots
- Bar charts
- Scatter plots
- Histograms
- Box plots
- Heatmaps



### Data Protection & Compliance

Document Anonymization: Automatically detect and redact sensitive information

- Personal names
- Addresses
- Phone numbers
- Email addresses
- ID numbers
- Dates
- Other PII (Personally Identifiable Information)



### Document Generation

- Custom Exports: Generate new documents based on analysis
- Multiple Format Support: Export to PDF, DOCX, or TXT
- Template Processing: Apply predefined templates to exported documents





## Tech Stack

DocMind AI is built on a modern tech stack:

**Backend** : FastAPI for high-performance API endpoints

**NLP Engine** : LangChain with Groq for advanced language understanding

**Document Processing** : Specialized tools for OCR, table extraction, and document parsing

**Visualization** : Matplotlib and Seaborn for data visualization

**Privacy Tools** : Custom anonymization pipeline using SpaCy NER and regex patterns


## Getting Started

**Prerequisites**

- Python 3.9+
- Tesseract OCR installed on your system
- Required Python packages (see requirements.txt)
- poppler-23.11.0 installed  on your system
## Installation

**Clone the repository**
```bash
git clone https://github.com/yourusername/docmind-ai.git
cd docmind-ai
```
**Create and activate a virtual environment**
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```
**Install dependencies**
```bash
pip install -r requirements.txt
```
**Install Ollama following https://ollama.com/ guide and run a model**

**Set up environment variables**
```bash
cp .env.example .env
# Edit .env with your API keys and configurations
```




    
## Environment Variables

To run this project, you will need to add the following environment variables to your .env file

`TESSERACT_CMD (eg:C:\Program Files\Tesseract-OCR\tesseract.exe)`


## Run Locally
Go to the project directory
```bash
python llmhackv1.py
```
The API will be available at http://localhost:8000


## API Reference

#### Load and process a document for analysis

```http
  POST /initialize
```

| Parameter | Type     | Description                |
| :-------- | :------- | :------------------------- |
| `session_id` | `string` | **Required**.  |
| `file_path` | `string` | **Required**.  |
| `question` | `string` | **Required**.  |

#### Submit questions or commands about the loaded document

```http
  POST /query
```

| Parameter | Type     | Description                       |
| :-------- | :------- | :-------------------------------- |
| `session_id` | `string` | **Required**.  |
| `question` | `string` | **Required**.  |




## Usage/Examples

```python
import requests

# Initialize a session with a document
response = requests.post(
    "http://localhost:8000/initialize",
    json={
        "file_path": "path/to/your/document.pdf",
        "question": "What is this document about?",
        "session_id": "user123"
    }
)

# Ask follow-up questions
response = requests.post(
    "http://localhost:8000/query",
    json={
        "question": "Can you summarize the key points?",
        "session_id": "user123"
    }
)
```


## Authors

- [Mohamed CAMARA](https://github.com/MohCw)
- [Mouhamed SARR](https://github.com/sarrmouhamed29)
- [Mohamed El Bachir TRAORE](https://github.com/BachirTra)
- [Magatte Taye Mbodj](https://github.com/magatte365)


## Contributing

Contributions are always welcome!

See `contributing.md` for ways to get started.

Please adhere to this project's `code of conduct`.

