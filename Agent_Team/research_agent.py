import os
from typing import Dict, List
from langchain.prompts import ChatPromptTemplate
from langchain_core.messages import SystemMessage
from langchain_openai import AzureChatOpenAI
from langchain.output_parsers import ResponseSchema, StructuredOutputParser
from langchain_community.tools import DuckDuckGoSearchRun
import json

search_tool = DuckDuckGoSearchRun()

class ResearcherAgent:
  def __init__(self, deployment_name=os.getenv("DEPLOYMENT_NAME")):
    self.llm = AzureChatOpenAI(
        deployment_name=deployment_name,
        openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
        azure_endpoint=os.getenv("AZURE_OPENAI_API_BASE"),
        openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
    )

    researcher_prompt = ChatPromptTemplate.from_messages(
        [SystemMessage(content="""You are a Researcher Agent that finds relevant information to answer specific questions.

        You'll receive a list of research questions along with context about a problem. Your job is to:
        1. Search for relevant information for each question
        2. Summarize the findings in a clear and concise way
        3. Cite sources when possible
        4. Organize the information to be useful for solving the original problem

        Your output should follow this format:
        ```
        {
        "research_findings": [
            {
            "question": "The original research question",
            "answer": "Your detailed answer",
            },
            ...
        ],
        "additional_insights": "Any important information you discovered that wasn't explicitly asked for"
        }
        ```"""
            ),
            ("human", "Context: {context}\nResearch Questions: {questions}"),
        ]
    )

    self.search_tool = DuckDuckGoSearchRun()

    researcher_output_schemas = [
        ResponseSchema(
            name="research_findings",
            description="List of answers to each research question with sources",
        ),
        ResponseSchema(
            name="additional_insights",
            description="Any additional important information discovered",
        ),
    ]

    self.output_parser = StructuredOutputParser.from_response_schemas(
        researcher_output_schemas
    )

    self.chain = researcher_prompt | self.llm | self.output_parser

  def run(self, context: str, questions: List[str]) -> Dict:
    """Run the researcher agent on a set of questions"""
    # First search for each question
    search_results = []
    for question in questions:
        search_result = self.search_tool.run(question)
        search_results.append(
          {"question": question, "search_result": search_result}
        )

    # Then analyze all search results together
    research_context = f"Problem context: {context}\n\nSearch results: {json.dumps(search_results, indent=2)}"
    result = self.chain.invoke(
      {"context": research_context, "questions": questions}
    )
    return result
