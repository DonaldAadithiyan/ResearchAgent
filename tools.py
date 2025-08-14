from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime

search = DuckDuckGoSearchRun()
wiki_wrapper = WikipediaAPIWrapper(top_k=5, lang="en")


wiki_tool = WikipediaQueryRun(api_wrapper=wiki_wrapper)

search_tool = Tool(
    name="search_tool",
    func=search.run,
    description="A tool to search the web for information. Use this tool when you need to find information that is not in the knowledge base. Input should be a search query.",
)