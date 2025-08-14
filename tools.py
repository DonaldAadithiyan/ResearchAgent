from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from langchain.memory import ConversationBufferMemory


search = DuckDuckGoSearchRun()
wiki_wrapper = WikipediaAPIWrapper(top_k=5, lang="en")


wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="A tool to search the web for information. Use this tool when you need to find information that is not in the knowledge base. Input should be a search query.",
)

def save_to_txt(data: str, filename: str = "research_output.txt"):
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    formatted_text = f"--- Research Output ---\nTimestamp: {timestamp}\n\n{data}\n\n"

    with open(filename, "a", encoding="utf-8") as f:
        f.write(formatted_text)
    
    return f"Data successfully saved to {filename}"

save_tool = Tool(
    name="save_text_to_file",
    func=save_to_txt,
    description="Saves structured research data to a text file.",
)

memory = ConversationBufferMemory(memory_key="chat_history", return_messages=True)
