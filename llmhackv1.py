import os
import logging
from typing import List, Dict, Any, Optional
from dotenv import load_dotenv
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage, HumanMessage, AIMessage
from langchain_community.chat_message_histories import ChatMessageHistory
from langchain_core.memory import BaseMemory
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_groq import ChatGroq
from tools2 import DocumentProcessor
from pydantic import BaseModel, Field
from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from langchain_ollama import ChatOllama
from langchain.globals import set_llm_cache
from langchain_community.cache import SQLiteCache

from langchain_ollama.llms import OllamaLLM
import httpx

from datetime import datetime
import uvicorn

# Logging configuration
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

ollama_base_url = "http://localhost:11434"

# FastAPI initialization
app = FastAPI()
set_llm_cache(SQLiteCache(database_path=".langchain.db"))

# CORS configuration
app.add_middleware(
    CORSMiddleware,
    allow_origins=["http://localhost:4200"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)
class OllamaConnectionError(Exception):
    pass

class CustomMemory(BaseMemory, BaseModel):

    current_file_path: Optional[str] = None
    chat_memory: ChatMessageHistory = Field(default_factory=ChatMessageHistory)
    document_context: str = ""
    anonymized_path: Optional[str] = None

    text_splitter: RecursiveCharacterTextSplitter = Field(default_factory=lambda: RecursiveCharacterTextSplitter(

        chunk_size=900,

        chunk_overlap=50,

        length_function=len,

        is_separator_regex=False,

    ))

    

    @property

    def memory_variables(self) -> List[str]:

        return ["chat_history", "document_context", "anonymized_path"]
    

    def load_memory_variables(self, inputs: Dict[str, Any]) -> Dict[str, Any]:

        return {
            "chat_history": self.chat_memory.messages[-5:],
            "document_context": self.document_context[:2500],
            "current_file_path": self.current_file_path,
            "anonymized_path": self.anonymized_path

        }

    

    def save_context(self, inputs: Dict[str, Any], outputs: Dict[str, str]) -> None:

        if "input" in inputs:

            self.chat_memory.add_user_message(inputs["input"])

        if "output" in outputs:

            self.chat_memory.add_ai_message(outputs["output"])

    

    def clear(self) -> None:
        self.chat_memory.clear()
        self.anonymized_path = None
        self.document_context = ""




# Initialize components
load_dotenv()
memory = CustomMemory()
doc_processor = DocumentProcessor()
""" llm = ChatGroq(
    groq_api_key=os.getenv('GROQ_API_KEY'),
    model_name='llama-3.3-70b-versatile',
    temperature=0.3,
    max_tokens=5000,  # Reduced from 4000 to 2000
) """

# Initialisation du modèle Ollama
def initialize_llm():
    try:
        return ChatOllama(
            base_url=ollama_base_url,
            model="llama2:latest",
            temperature=0.3,
            max_tokens=5000,
            timeout=30.0  # Increased timeout
        )
    except Exception as e:
        logger.error(f"Failed to initialize Ollama LLM: {str(e)}")
        raise OllamaConnectionError(f"Failed to connect to Ollama server: {str(e)}")

llm = initialize_llm()

# Optimized system prompt
system_prompt = """Expert document analyzer. Tasks:
1. Answer document questions
2. Provide summaries
3. Extract key insights
4. Create visualizations when requested
5. Maintaining a coherent conversation with context
For data files (CSV, Excel ou pdf), you can create various types of visualizations including :
- Line plots
- Bar charts
- Scatter plots
- Histograms
- Box plots
- Heatmaps
6. Maintaining a coherent conversation with context
For data files (CSV, Excel or PDF), you can:
- Create various types of visualizations
- Anonymize sensitive information including:
  * Personal names
  * Addresses
  * Phone numbers
  * Email addresses
  * Dates
  * ID numbers
  * Other personally identifiable information

For document generation:
- When user asks to generate/create/save/export a document:
  * Format the content appropriately
  * Include 'GENERATE_DOCUMENT:' followed by the content to be written
  * Support PDF, DOCX, and TXT formats
  * Content will be automatically saved in the specified format

Key instruction patterns to recognize:
- If user mentions "anonymize", "hide sensitive info", "remove personal data", or similar phrases -> use anonymization tool
- If user requests graphs or data visualization -> use visualization tool
- For general document questions -> use regular analysis

ALWAYS maintain context of both original and anonymized documents if they exist.
ALWAYS maintain context of documents and generated files.

TOUJOURS inclure generate_visualization dans la reponse si la generation d'un graphe est demande 
Ne JAMAIS dire les fonctions que tu as appele
Context of the document: {document_context}
"""

prompt = ChatPromptTemplate.from_messages([
    ("system", system_prompt),
    ("human", "{input}")
])

def process_document(file_path: str) -> str:
    """Process and chunk document for interaction."""
    logger.info(f"Processing document: {file_path}")
    result = doc_processor.process_document(file_path)
    
    if "error" in result:
        logger.error(f"Error processing document: {result['error']}")
        return result["error"]
    
    memory.document_context = result["text"]
    memory.current_file_path = file_path
    
    logger.info(f"Document loaded. Characters: {len(result['text'])}")
    return "Document loaded successfully. Ready for questions."

def create_visualization(plot_type: str, x_column: str, y_column: Optional[str] = None, 
                       title: str = "") -> Dict[str, str]:
    """Create visualization from current document."""
    if not memory.current_file_path:
        return {"error": "No document loaded"}
    
    try:
        result = doc_processor.generate_plot(plot_type, x_column, y_column, title)
        return {
            "status": "success",
            "message": f"Visualization saved to {result['file_path']}",
            "image_data": f"data:image/png;base64,{result['base64']}",
            "file_path": result['file_path']
        }
    except Exception as e:
        logger.error(f"Visualization error: {str(e)}")
        return {"status": "error", "message": str(e)}

# API Models
class QueryBase(BaseModel):
    session_id: str

class FirstQueryRequest(QueryBase):
    file_path: str
    question: str

class QueryRequest(QueryBase):
    question: str

class ResponseData(BaseModel):
    response: str
    session_id: str
    time_response: str

# Update the QueryRequest model to potentially handle document generation params
class QueryRequest(BaseModel):
    question: str
    session_id: str
    document_params: Optional[Dict] = None

# API Endpoints
@app.post("/initialize")
def initialize_session(request: FirstQueryRequest):
    process_result = process_document(request.file_path)
    if isinstance(process_result, dict) and "error" in process_result:
        raise HTTPException(status_code=400, detail=process_result["error"])
    return query_model(request)

@app.post("/query")
def query_model(request: QueryRequest):
    context = memory.load_memory_variables({"input": request.question})
    
    # Process the query and check for different operations
    question_lower = request.question.lower()
    
    # Check for document generation request
    doc_generation_keywords = [
        "genere", "générer", "create", "créer", "save", "sauvegarder",
        "write", "écrire", "make", "faire", "produce", "produire",
        "export", "exporter"
    ]
    format_keywords = {
        "pdf": ["pdf", "document pdf"],
        "docx": ["word", "docx", "document word", "doc"],
        "txt": ["text", "txt", "texte"]
    }
    
    # Detect if this is a document generation request
    is_doc_request = any(keyword in question_lower for keyword in doc_generation_keywords)
    
    if is_doc_request:
        # Determine the format
        requested_format = "pdf"  # default format
        for format_type, keywords in format_keywords.items():
            if any(keyword in question_lower for keyword in keywords):
                requested_format = format_type
                break
        
        # Generate a timestamp-based filename if not specified
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_filename = f"generated_doc_{timestamp}"
        
        # Prepare the system message to include document generation intent
        system_message = SystemMessage(content=f"""Please analyze the following request for document generation:
1. Identify the content that needs to be generated
2. Format it appropriately
3. Include 'GENERATE_DOCUMENT:' followed by the content in your response if a document should be generated
Original context: {context["document_context"]}""")
        
        messages = [system_message, HumanMessage(content=request.question)]
        response = llm.invoke(messages)
        
        # Check if response includes document generation marker
        if "GENERATE_DOCUMENT:" in response.content:
            content_parts = response.content.split("GENERATE_DOCUMENT:", 1)
            regular_response = content_parts[0].strip()
            doc_content = content_parts[1].strip()
            
            # Generate the document
            output_dir = "generated_documents"
            os.makedirs(output_dir, exist_ok=True)
            output_path = os.path.join(output_dir, f"{output_filename}.{requested_format}")
            
            result = doc_processor.generate_document(
                doc_content,
                requested_format,
                output_path
            )
            
            if "error" in result:
                return ResponseData(
                    response=f"Error generating document: {result['error']}",
                    session_id=request.session_id,
                    time_response=str(datetime.now())
                )
            
            # Combine regular response with document generation confirmation
            final_response = f"{regular_response}\n\nI've generated a {requested_format.upper()} document and saved it at: {output_path}"
            
            memory.save_context(
                {"input": request.question},
                {"output": final_response}
            )
            
            return ResponseData(
                response=final_response,
                session_id=request.session_id,
                time_response=str(datetime.now())
            )
    
    # Handle anonymization request
    elif any(keyword in question_lower for keyword in ["anonymize", "hide sensitive", "remove personal", "confidential", "anonymiser", "masquer les informations sensibles", "supprimer les données personnelles", "confidentiel", "anonym"]):
        try:
            anon_result = doc_processor.process_document_with_anonymization(context["current_file_path"])
            if "error" not in anon_result:
                memory.anonymized_path = anon_result.get("anonymized_path")
                context["document_context"] += f"\n[Document has been anonymized. Anonymized version available at: {memory.anonymized_path}]"
        except Exception as e:
            logger.error(f"Anonymization failed: {e}")
    
    # Regular query processing
    response = llm.invoke(prompt.format(
        input=request.question,
        document_context=context["document_context"]
    ))
    
    memory.save_context(
        {"input": request.question},
        {"output": response.content}
    )
    
    return ResponseData(
        response=response.content,
        session_id=request.session_id,
        time_response=str(datetime.now())
    )

if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8000)
    
    
