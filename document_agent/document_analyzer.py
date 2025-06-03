from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
from langchain.chains import RetrievalQA
import os
from dotenv import load_dotenv
import PyPDF2
import re

# Load environment variables
load_dotenv()

class DocumentAnalysisAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
        
    def load_document(self, file_path: str) -> str:
        """Load document content based on file type."""
        if file_path.endswith('.pdf'):
            return self._read_pdf(file_path)
        elif file_path.endswith('.txt'):
            return self._read_text(file_path)
        else:
            raise ValueError("Unsupported file type")
    
    def _read_pdf(self, file_path: str) -> str:
        """Extract text from PDF file."""
        with open(file_path, 'rb') as file:
            pdf_reader = PyPDF2.PdfReader(file)
            text = ""
            for page in pdf_reader.pages:
                text += page.extract_text()
        return text
    
    def _read_text(self, file_path: str) -> str:
        """Read text file."""
        with open(file_path, 'r', encoding='utf-8') as file:
            return file.read()
    
    def create_vector_store(self, text: str) -> FAISS:
        """Create vector store from document text."""
        texts = self.text_splitter.split_text(text)
        return FAISS.from_texts(texts, self.embeddings)
    
    def analyze_document(self, file_path: str, query: str = None) -> Dict[str, Any]:
        """Analyze document and return insights."""
        # Load and process document
        text = self.load_document(file_path)
        vector_store = self.create_vector_store(text)
        
        # Create QA chain
        qa_chain = RetrievalQA.from_chain_type(
            llm=self.llm,
            chain_type="stuff",
            retriever=vector_store.as_retriever()
        )
        
        # Generate summary
        summary_prompt = f"Please provide a comprehensive summary of the following text:\n\n{text[:2000]}..."
        summary = self.llm.predict(summary_prompt)
        
        # Extract key points
        key_points_prompt = f"Extract the 5 most important points from this text:\n\n{text[:2000]}..."
        key_points = self.llm.predict(key_points_prompt)
        
        # Answer specific query if provided
        query_answer = None
        if query:
            query_answer = qa_chain.run(query)
        
        return {
            "summary": summary,
            "key_points": key_points,
            "query_answer": query_answer if query else "No query provided"
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = DocumentAnalysisAgent()
    
    # Example document analysis
    try:
        results = agent.analyze_document(
            file_path="path/to/your/document.pdf",
            query="What are the main findings?"
        )
        
        print("Document Analysis Results:")
        print("\nSummary:")
        print(results["summary"])
        print("\nKey Points:")
        print(results["key_points"])
        print("\nQuery Answer:")
        print(results["query_answer"])
        
    except Exception as e:
        print(f"Error during document analysis: {str(e)}") 