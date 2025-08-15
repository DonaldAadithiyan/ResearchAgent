from dotenv import load_dotenv
from pydantic import BaseModel
from langchain_openai import ChatOpenAI
from langchain_anthropic import ChatAnthropic
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import PydanticOutputParser
from langchain.agents import create_tool_calling_agent, AgentExecutor
from tools import search_tool, wiki_tool, save_tool, memory, analyze_tool
from langchain.schema import HumanMessage
import pyfiglet 
from colorama import init, Fore, Style

load_dotenv()
init(autoreset=True)

class AgentResponse(BaseModel):
    topic: str
    summary: str
    sources: list[str]
    tools_used: list[str]

llm = ChatOpenAI(model="gpt-4o-mini", temperature=0.0)
llm2 = ChatAnthropic(model="claude-2", temperature=0.0)

parser = PydanticOutputParser(pydantic_object=AgentResponse)

prompt = ChatPromptTemplate.from_messages(
    [
        (
            "system",
            """
            You are a research assistant that will help generate a research paper.
            Answer the user query and use neccessary tools. 
            Wrap the output in this format and provide no other text\n{format_instructions}
            """,
        ),
        ("placeholder", "{chat_history}"),
        ("human", "{query}"),
        ("placeholder", "{agent_scratchpad}"),
    ]
).partial(format_instructions=parser.get_format_instructions())   

history = memory.load_memory_variables({})["chat_history"]



tools = [search_tool, wiki_tool, save_tool, analyze_tool]

agent = create_tool_calling_agent(
    llm=llm,
    tools=tools,
    prompt=prompt
)

agent_executor = AgentExecutor(
    agent=agent,
    tools=tools,
    memory=memory,
    verbose=True)
# query = input("Enter your research query: ")

# Create big ASCII text
ascii_banner = pyfiglet.figlet_format("Research Agent")
print(Fore.CYAN + ascii_banner)

# Add a colored instruction below
print(Fore.YELLOW + Style.BRIGHT + "Type 'exit' to quit the program.\n")
print(Fore.YELLOW + Style.BRIGHT + "Type 'analyze' to analyze a CSV\n")

while True:
    query = input(Fore.GREEN + "Enter your research query: ").strip()
    if query.lower() == "exit":
        print("Exiting Research Agent.")
        break
    elif query.lower() == "analyze":
        file_path = input(Fore.GREEN + "Enter the path to the CSV file: ").strip()
        if not file_path:
            print(Fore.RED + "No file path provided. Please try again.")
            continue
        
        # Invoke analyze tool
        try:
            response = analyze_tool.func(file_path)
            print(Fore.BLUE + "Analysis Result:\n", response)
        except Exception as e:
            print(Fore.RED + f"Error analyzing CSV: {e}")
        continue

    # Get chat history
    history = memory.load_memory_variables({})["chat_history"]
    messages = history + [HumanMessage(content=query)]

    # Invoke agent
    try:
        raw_response = agent_executor.invoke({"query": query})

        output_text = raw_response.get("output")
        if isinstance(output_text, list):
            output_text = output_text[0]

        structured_response = parser.parse(output_text)
        print("\nStructured Response:\n", structured_response)
    except Exception as e:
        print(f"Error parsing response: {e}")
