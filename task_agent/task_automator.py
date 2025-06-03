from typing import List, Dict, Any, Callable
from langchain.agents import Tool, AgentExecutor, LLMSingleActionAgent
from langchain.prompts import StringPromptTemplate
from langchain.chat_models import ChatOpenAI
from langchain.schema import AgentAction, AgentFinish
from langchain.tools import BaseTool
import os
from dotenv import load_dotenv
import json
import time
from datetime import datetime
import logging
from pathlib import Path
import schedule
import requests
import smtplib
from email.mime.text import MIMEText
from email.mime.multipart import MIMEMultipart

# Load environment variables
load_dotenv()

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler('task_automation.log'),
        logging.StreamHandler()
    ]
)

class TaskAutomationAgent:
    def __init__(self, api_key: str = None):
        self.api_key = api_key or os.getenv("OPENAI_API_KEY")
        self.llm = ChatOpenAI(temperature=0, model_name="gpt-4")
        self.tasks = {}
        self.task_history = []
        self.logger = logging.getLogger(__name__)
    
    def register_task(self, name: str, task_func: Callable, schedule_time: str = None) -> None:
        """Register a new task with optional scheduling."""
        self.tasks[name] = {
            "function": task_func,
            "schedule": schedule_time,
            "last_run": None,
            "status": "registered"
        }
        
        if schedule_time:
            schedule.every().day.at(schedule_time).do(self.execute_task, name)
    
    def execute_task(self, task_name: str, *args, **kwargs) -> Dict[str, Any]:
        """Execute a registered task."""
        if task_name not in self.tasks:
            raise ValueError(f"Task '{task_name}' not found")
        
        task = self.tasks[task_name]
        start_time = time.time()
        
        try:
            self.logger.info(f"Starting task: {task_name}")
            result = task["function"](*args, **kwargs)
            
            # Update task status
            task["last_run"] = datetime.now()
            task["status"] = "completed"
            
            # Record in history
            self.task_history.append({
                "task_name": task_name,
                "start_time": start_time,
                "end_time": time.time(),
                "status": "success",
                "result": result
            })
            
            self.logger.info(f"Task completed: {task_name}")
            return {"status": "success", "result": result}
            
        except Exception as e:
            self.logger.error(f"Task failed: {task_name} - {str(e)}")
            
            # Update task status
            task["status"] = "failed"
            
            # Record in history
            self.task_history.append({
                "task_name": task_name,
                "start_time": start_time,
                "end_time": time.time(),
                "status": "failed",
                "error": str(e)
            })
            
            return {"status": "failed", "error": str(e)}
    
    def monitor_tasks(self) -> None:
        """Monitor and run scheduled tasks."""
        while True:
            schedule.run_pending()
            time.sleep(60)
    
    def get_task_status(self, task_name: str = None) -> Dict[str, Any]:
        """Get status of specific task or all tasks."""
        if task_name:
            if task_name not in self.tasks:
                raise ValueError(f"Task '{task_name}' not found")
            return self.tasks[task_name]
        return self.tasks
    
    def get_task_history(self, task_name: str = None) -> List[Dict[str, Any]]:
        """Get history of specific task or all tasks."""
        if task_name:
            return [h for h in self.task_history if h["task_name"] == task_name]
        return self.task_history
    
    def send_notification(self, subject: str, message: str, recipients: List[str]) -> None:
        """Send email notification."""
        try:
            # Configure email settings
            sender_email = os.getenv("EMAIL_USER")
            sender_password = os.getenv("EMAIL_PASSWORD")
            smtp_server = os.getenv("SMTP_SERVER", "smtp.gmail.com")
            smtp_port = int(os.getenv("SMTP_PORT", "587"))
            
            # Create message
            msg = MIMEMultipart()
            msg["From"] = sender_email
            msg["To"] = ", ".join(recipients)
            msg["Subject"] = subject
            
            msg.attach(MIMEText(message, "plain"))
            
            # Send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.starttls()
                server.login(sender_email, sender_password)
                server.send_message(msg)
            
            self.logger.info(f"Notification sent to {recipients}")
            
        except Exception as e:
            self.logger.error(f"Failed to send notification: {str(e)}")
            raise

# Example task functions
def example_file_operation(file_path: str, content: str) -> Dict[str, Any]:
    """Example task: Write content to a file."""
    try:
        with open(file_path, 'w') as f:
            f.write(content)
        return {"message": f"Content written to {file_path}"}
    except Exception as e:
        raise Exception(f"File operation failed: {str(e)}")

def example_api_call(url: str) -> Dict[str, Any]:
    """Example task: Make an API call."""
    try:
        response = requests.get(url)
        response.raise_for_status()
        return response.json()
    except Exception as e:
        raise Exception(f"API call failed: {str(e)}")

# Example usage
if __name__ == "__main__":
    # Initialize the agent
    agent = TaskAutomationAgent()
    
    try:
        # Register tasks
        agent.register_task(
            "file_operation",
            example_file_operation,
            schedule_time="10:00"
        )
        
        agent.register_task(
            "api_call",
            example_api_call,
            schedule_time="15:00"
        )
        
        # Execute tasks immediately
        file_result = agent.execute_task(
            "file_operation",
            file_path="example.txt",
            content="Hello, World!"
        )
        print("File operation result:", file_result)
        
        api_result = agent.execute_task(
            "api_call",
            url="https://api.example.com/data"
        )
        print("API call result:", api_result)
        
        # Get task status
        status = agent.get_task_status()
        print("\nTask Status:")
        print(json.dumps(status, indent=2))
        
        # Get task history
        history = agent.get_task_history()
        print("\nTask History:")
        print(json.dumps(history, indent=2))
        
        # Send notification
        agent.send_notification(
            subject="Task Automation Report",
            message="Tasks completed successfully",
            recipients=["user@example.com"]
        )
        
    except Exception as e:
        print(f"Error during task automation: {str(e)}") 