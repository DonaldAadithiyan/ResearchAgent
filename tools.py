from langchain_community.tools import WikipediaQueryRun, DuckDuckGoSearchRun
from langchain_community.utilities import WikipediaAPIWrapper
from langchain.tools import Tool
from datetime import datetime
from langchain.memory import ConversationBufferMemory
import os
import pandas as pd
import matplotlib.pyplot as plt


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

def analyze_csv(file_path: str) -> str:
    """Analyze a time-series CSV file and return trends, anomalies, and summary stats."""
    if not os.path.exists(file_path):
        return f"Error: File '{file_path}' not found."

    try:
        # Load CSV
        df = pd.read_csv(file_path)

        # Detect time column
        time_col = None
        for col in df.columns:
            try:
                df[col] = pd.to_datetime(df[col], errors='coerce')
                if df[col].notnull().sum() > len(df) * 0.5:
                    time_col = col
                    break
            except Exception:
                continue

        if time_col:
            df = df.set_index(time_col).sort_index()

        # Basic summary
        summary = []
        summary.append(f"Rows: {len(df)}, Columns: {df.columns.tolist()}")
        summary.append("\nDescriptive stats:\n" + str(df.describe()))

        # Monthly trends if time column found
        if time_col:
            monthly_mean = df.resample('M').mean(numeric_only=True)
            summary.append("\nMonthly mean values (first 5 rows):\n" + str(monthly_mean.head()))

        # Save a trend plot
        numeric_cols = df.select_dtypes(include='number').columns
        if len(numeric_cols) > 0:
            plt.figure(figsize=(10,5))
            df[numeric_cols[0]].plot(title=f"Trend of {numeric_cols[0]}")
            plot_path = "trend.png"
            plt.savefig(plot_path)
            summary.append(f"\nTrend plot saved to {plot_path}")

        return "\n".join(summary)

    except Exception as e:
        return f"Error analyzing CSV: {str(e)}"
    
analyze_tool = Tool(
    name="analyze_csv",
    func=analyze_csv,
    description="Analyzes a time-series CSV file and returns trends, anomalies, and summary statistics. Input should be the file path of the CSV.",
)