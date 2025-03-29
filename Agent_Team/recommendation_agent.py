import os
from typing import Dict
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain_core.output_parsers import StrOutputParser
import json

class RecommendationAgent:
    def __init__(self, deployment_name=os.getenv("DEPLOYMENT_NAME")):
        self.llm = AzureChatOpenAI(
            deployment_name=deployment_name,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
        
        recommendation_prompt = ChatPromptTemplate.from_messages([
            SystemMessage(content="""You are a Recommendation Agent that creates comprehensive, detailed implementation plans in beautifully formatted markdown.
                You'll receive a problem description, a solution schema, and research findings. Your job is to:
                Create a detailed, structured implementation report with clear sections, examples, and actionable steps in markdown format.

                Format your response EXACTLY like this:
                
					### Executive Summary
					Brief overview...
					
					### Detailed Implementation Plan
					
					#### Section 1
					**Objective:**...
					**Key Components:**...
					**Implementation Details:**...
					**Challenges & Solutions:**...
					---
					[More sections...]
					
					### Tools & Resources
					- **Tool**: Description...
					
					### Key Risks & Mitigations
					- **Risk**: Mitigation...
					
					### Expected Outcomes
					- Outcome 1...
					
					### Next Steps
					1. Step 1...
                
                Important Formatting Notes: 
                - Use ONLY MARKDOWN FORMATTING, no JSON or code blocks
                - Never wrap the response in ```markdown tags
                - Maintain the exact header structure"""),
            ("human", "Problem: {problem}\nSolution Schema: {schema}\nResearch Findings: {findings}"),
        ])
        
        self.chain = recommendation_prompt | self.llm | StrOutputParser()
    
    def run(self, problem: str, schema: Dict, findings: Dict) -> str:
        """Run the recommendation agent and return markdown text"""
        return self.chain.invoke({
            "problem": problem,
            "schema": json.dumps(schema, indent=2),
            "findings": json.dumps(findings, indent=2)
        })