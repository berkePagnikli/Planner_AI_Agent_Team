import os
from abc import ABC, abstractmethod
from typing import Dict, Any
from langchain.prompts import ChatPromptTemplate
from langchain_openai import AzureChatOpenAI


class Agent(ABC):
    """Base class for all agents with common Azure OpenAI configuration and system prompt setup."""
    
    def __init__(self):
        """Initialize the agent with Azure OpenAI configuration."""
        
        self.deployment_name = os.getenv("DEPLOYMENT_NAME")
        self.llm = self._create_llm()
        self.system_prompt = self._get_system_prompt()
        self.prompt_template = self._create_prompt_template()
        self.output_parser = self._create_output_parser()
        self.chain = self._create_chain()
    
    def _create_llm(self) -> AzureChatOpenAI:
        """Create and configure the Azure OpenAI LLM instance."""
        return AzureChatOpenAI(
            deployment_name=self.deployment_name,
            openai_api_version=os.getenv("AZURE_OPENAI_API_VERSION"),
            azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT"),
            openai_api_key=os.getenv("AZURE_OPENAI_API_KEY"),
        )
    
    @abstractmethod
    def _get_system_prompt(self) -> str:
        """Get the system prompt for this agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_prompt_template(self) -> ChatPromptTemplate:
        """Create the prompt template for this agent. Must be implemented by subclasses."""
        pass
    
    @abstractmethod
    def _create_output_parser(self) -> Any:
        """Create the output parser for this agent. Must be implemented by subclasses."""
        pass
    
    def _create_chain(self) -> Any:
        """Create the LLM chain. Can be overridden by subclasses if needed."""
        return self.prompt_template | self.llm | self.output_parser
    
    @abstractmethod
    def run(self, *args, **kwargs) -> Any:
        """Execute the agent's main functionality. Must be implemented by subclasses."""
        pass
