from typing import List, Dict, Any
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
from langchain.utilities import GoogleSerperAPIWrapper
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.embeddings import OpenAIEmbeddings
from langchain.vectorstores import FAISS
import os
from dotenv import load_dotenv
import requests
from bs4 import BeautifulSoup
import json

# Load environment variables
load_dotenv()

class WebResearchAgent:
    def __init__(self, api_key: str = None, serper_api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.serper_api_key = serper_api_key or os.getenv("SERPER_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.search = GoogleSerperAPIWrapper()
        self.embeddings = OpenAIEmbeddings()
        self.text_splitter = RecursiveCharacterTextSplitter(
            chunk_size=1000,
            chunk_overlap=200
        )
    
    def search_web(self, query: str, num_results: int = 5) -> List[Dict[str, Any]]:
        """Search the web for information."""
        search_results = self.search.results(query, num_results)
        return search_results
    
    def extract_content(self, url: str) -> str:
        """Extract content from a webpage."""
        try:
            response = requests.get(url, timeout=10)
            response.raise_for_status()
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Remove script and style elements
            for script in soup(["script", "style"]):
                script.decompose()
            
            # Get text content
            text = soup.get_text()
            
            # Clean up text
            lines = (line.strip() for line in text.splitlines())
            chunks = (phrase.strip() for line in lines for phrase in line.split("  "))
            text = ' '.join(chunk for chunk in chunks if chunk)
            
            return text
        except Exception as e:
            return f"Error extracting content: {str(e)}"
    
    def analyze_content(self, content: str) -> Dict[str, Any]:
        """Analyze the content and extract key information."""
        # Create vector store
        texts = self.text_splitter.split_text(content)
        vector_store = FAISS.from_texts(texts, self.embeddings)
        
        # Generate summary
        summary_prompt = f"Please provide a comprehensive summary of the following text:\n\n{content[:2000]}..."
        summary = self.llm.predict(summary_prompt)
        
        # Extract key points
        key_points_prompt = f"Extract the 5 most important points from this text:\n\n{content[:2000]}..."
        key_points = self.llm.predict(key_points_prompt)
        
        return {
            "summary": summary,
            "key_points": key_points
        }
    
    def research_topic(self, topic: str, num_sources: int = 3) -> Dict[str, Any]:
        """Conduct comprehensive research on a topic."""
        # Search for relevant information
        search_results = self.search_web(topic, num_sources)
        
        # Extract and analyze content from each source
        sources_analysis = []
        for result in search_results:
            if 'link' in result:
                content = self.extract_content(result['link'])
                analysis = self.analyze_content(content)
                sources_analysis.append({
                    "url": result['link'],
                    "title": result.get('title', ''),
                    "analysis": analysis
                })
        
        # Synthesize findings
        synthesis_prompt = f"""Based on the following analyses from multiple sources, 
        provide a comprehensive synthesis of the information about {topic}:
        
        {json.dumps(sources_analysis, indent=2)}
        """
        
        synthesis = self.llm.predict(synthesis_prompt)
        
        return {
            "topic": topic,
            "sources_analyzed": len(sources_analysis),
            "sources": sources_analysis,
            "synthesis": synthesis
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = WebResearchAgent()
    
    # Example research
    try:
        results = agent.research_topic(
            topic="Latest developments in quantum computing",
            num_sources=3
        )
        
        print("Research Results:")
        print(f"\nTopic: {results['topic']}")
        print(f"Sources Analyzed: {results['sources_analyzed']}")
        print("\nSynthesis:")
        print(results['synthesis'])
        
        print("\nDetailed Source Analysis:")
        for source in results['sources']:
            print(f"\nSource: {source['title']}")
            print(f"URL: {source['url']}")
            print("\nSummary:")
            print(source['analysis']['summary'])
            print("\nKey Points:")
            print(source['analysis']['key_points'])
            
    except Exception as e:
        print(f"Error during research: {str(e)}") 