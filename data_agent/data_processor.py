from typing import List, Dict, Any, Union
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
import pandas as pd
import numpy as np
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns

# Load environment variables
load_dotenv()

class DataProcessingAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
    
    def load_data(self, file_path: str) -> pd.DataFrame:
        """Load data from various file formats."""
        file_extension = Path(file_path).suffix.lower()
        
        if file_extension == '.csv':
            return pd.read_csv(file_path)
        elif file_extension in ['.xlsx', '.xls']:
            return pd.read_excel(file_path)
        elif file_extension == '.json':
            return pd.read_json(file_path)
        else:
            raise ValueError(f"Unsupported file format: {file_extension}")
    
    def analyze_data(self, df: pd.DataFrame) -> Dict[str, Any]:
        """Perform comprehensive data analysis."""
        analysis = {
            "basic_info": {
                "shape": df.shape,
                "columns": df.columns.tolist(),
                "dtypes": df.dtypes.astype(str).to_dict(),
                "missing_values": df.isnull().sum().to_dict()
            },
            "statistics": df.describe().to_dict(),
            "correlations": df.select_dtypes(include=[np.number]).corr().to_dict()
        }
        
        # Generate insights
        insights_prompt = f"""Based on the following data analysis, provide:
        1. Key observations about the data
        2. Potential data quality issues
        3. Interesting patterns or relationships
        4. Recommendations for further analysis
        
        Analysis results:
        {json.dumps(analysis, indent=2)}
        """
        
        insights = self.llm.predict(insights_prompt)
        analysis["insights"] = insights
        
        return analysis
    
    def clean_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """Clean and preprocess the data."""
        # Create a copy to avoid modifying the original
        df_clean = df.copy()
        
        # Handle missing values
        for col in df_clean.columns:
            if df_clean[col].dtype in [np.float64, np.int64]:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        
        # Remove duplicates
        df_clean.drop_duplicates(inplace=True)
        
        # Handle outliers for numerical columns
        for col in df_clean.select_dtypes(include=[np.number]).columns:
            Q1 = df_clean[col].quantile(0.25)
            Q3 = df_clean[col].quantile(0.75)
            IQR = Q3 - Q1
            df_clean = df_clean[
                (df_clean[col] >= Q1 - 1.5 * IQR) & 
                (df_clean[col] <= Q3 + 1.5 * IQR)
            ]
        
        return df_clean
    
    def generate_visualizations(self, df: pd.DataFrame, output_dir: str) -> Dict[str, str]:
        """Generate and save various visualizations."""
        os.makedirs(output_dir, exist_ok=True)
        visualization_paths = {}
        
        # Set style
        plt.style.use('seaborn')
        
        # Numerical columns distributions
        for col in df.select_dtypes(include=[np.number]).columns:
            plt.figure(figsize=(10, 6))
            sns.histplot(data=df, x=col)
            plt.title(f'Distribution of {col}')
            path = os.path.join(output_dir, f'{col}_distribution.png')
            plt.savefig(path)
            plt.close()
            visualization_paths[f'{col}_distribution'] = path
        
        # Correlation heatmap
        plt.figure(figsize=(12, 8))
        sns.heatmap(df.select_dtypes(include=[np.number]).corr(), annot=True, cmap='coolwarm')
        plt.title('Correlation Heatmap')
        path = os.path.join(output_dir, 'correlation_heatmap.png')
        plt.savefig(path)
        plt.close()
        visualization_paths['correlation_heatmap'] = path
        
        # Categorical columns
        for col in df.select_dtypes(include=['object']).columns:
            plt.figure(figsize=(10, 6))
            df[col].value_counts().plot(kind='bar')
            plt.title(f'Distribution of {col}')
            plt.xticks(rotation=45)
            path = os.path.join(output_dir, f'{col}_distribution.png')
            plt.savefig(path)
            plt.close()
            visualization_paths[f'{col}_distribution'] = path
        
        return visualization_paths
    
    def process_data(self, file_path: str, output_dir: str) -> Dict[str, Any]:
        """Complete data processing pipeline."""
        # Load data
        df = self.load_data(file_path)
        
        # Analyze original data
        original_analysis = self.analyze_data(df)
        
        # Clean data
        df_clean = self.clean_data(df)
        
        # Analyze cleaned data
        cleaned_analysis = self.analyze_data(df_clean)
        
        # Generate visualizations
        visualization_paths = self.generate_visualizations(df_clean, output_dir)
        
        # Generate report
        report_prompt = f"""Based on the data processing results, provide a comprehensive report:
        1. Data quality assessment
        2. Key findings and insights
        3. Impact of data cleaning
        4. Recommendations for further analysis
        
        Original Analysis:
        {json.dumps(original_analysis, indent=2)}
        
        Cleaned Analysis:
        {json.dumps(cleaned_analysis, indent=2)}
        """
        
        report = self.llm.predict(report_prompt)
        
        return {
            "original_analysis": original_analysis,
            "cleaned_analysis": cleaned_analysis,
            "visualization_paths": visualization_paths,
            "report": report
        }

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = DataProcessingAgent()
    
    # Example data processing
    try:
        results = agent.process_data(
            file_path="path/to/your/data.csv",
            output_dir="path/to/output"
        )
        
        print("Data Processing Results:")
        print("\nReport:")
        print(results["report"])
        
        print("\nVisualization Paths:")
        for name, path in results["visualization_paths"].items():
            print(f"{name}: {path}")
        
    except Exception as e:
        print(f"Error during data processing: {str(e)}") 