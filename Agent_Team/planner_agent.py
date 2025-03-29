import os
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser

class PlannerAgent:
    def __init__(self, deployment_name=os.getenv("DEPLOYMENT_NAME")):
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        planner_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Planning Agent that creates detailed solution schemas.
            
            You analyze user requests and break them down into actionable steps. Your job is to:
            1. Understand the user's problem
            2. Create a step-by-step plan to solve it
            3. Design a schema that represents the solution approach
            4. Specify what information needs to be researched
            
            Your output should follow this format:
            ```
            {
              "problem_understanding": "Brief analysis of the user's problem",
              "solution_schema": {
                "steps": ["Step 1", "Step 2", "..."],
                "components": ["Component 1", "Component 2", "..."]
              },
              "research_questions": ["Question 1", "Question 2", "..."]
            }
            ```"""),
            MessagesPlaceholder(variable_name="history"),
            ("human", "{input}"),
        ])
        
        planner_output_schemas = [
            ResponseSchema(name="problem_understanding", description="Brief analysis of the user's problem"),
            ResponseSchema(name="solution_schema", description="Object containing steps and components for the solution"),
            ResponseSchema(name="research_questions", description="List of questions that need to be researched")
        ]
        
        self.output_parser = StructuredOutputParser.from_response_schemas(planner_output_schemas)
        
        self.chain = planner_prompt | self.llm | self.output_parser
    
    def run(self, query: str, history: List = []) -> Dict:
        """Run the planner agent on a query"""
        result = self.chain.invoke({"input": query, "history": history})
        return result