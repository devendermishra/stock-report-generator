"""
Example of using LangChain tools with agents.
This demonstrates how the converted tools can be used by LLMs.
"""

from langchain.agents import create_openai_functions_agent, AgentExecutor
from langchain_openai import ChatOpenAI
from langchain.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain.schema import HumanMessage

from ..tools.langchain_tools import get_all_tools, initialize_tools, TOOL_DESCRIPTIONS

def create_stock_analysis_agent(openai_api_key: str):
    """
    Create a LangChain agent that can use all the stock analysis tools.
    
    Args:
        openai_api_key: OpenAI API key
        
    Returns:
        Configured LangChain agent
    """
    # Initialize tools
    initialize_tools(openai_api_key)
    
    # Get all available tools
    tools = get_all_tools()
    
    # Create the LLM
    llm = ChatOpenAI(
        model="gpt-4o-mini",
        temperature=0.1,
        api_key=openai_api_key
    )
    
    # Create the prompt template
    prompt = ChatPromptTemplate.from_messages([
        ("system", """You are a financial analyst expert. You have access to various tools for:
        - Getting stock data and company information
        - Searching for news and market trends
        - Summarizing and analyzing text content
        
        Use these tools to provide comprehensive analysis when requested.
        Always explain your reasoning and cite sources when possible."""),
        ("user", "{input}"),
        MessagesPlaceholder(variable_name="agent_scratchpad"),
    ])
    
    # Create the agent
    agent = create_openai_functions_agent(llm, tools, prompt)
    
    # Create the agent executor
    agent_executor = AgentExecutor(
        agent=agent,
        tools=tools,
        verbose=True,
        handle_parsing_errors=True,
        max_iterations=10
    )
    
    return agent_executor

def example_usage():
    """Example of how to use the agent."""
    # This would be called with a real API key
    # agent = create_stock_analysis_agent("your-openai-api-key")
    
    # Example queries the agent could handle:
    example_queries = [
        "Get stock metrics for RELIANCE and analyze the company's financial health",
        "Search for recent news about the banking sector in India",
        "Summarize the latest quarterly results for TCS and extract key insights",
        "Compare the performance of HDFC Bank and ICICI Bank",
        "Find market trends for the pharmaceutical sector"
    ]
    
    print("Example queries the agent can handle:")
    for i, query in enumerate(example_queries, 1):
        print(f"{i}. {query}")

if __name__ == "__main__":
    example_usage()
