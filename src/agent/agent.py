from langchain.agents import create_agent
from langchain_groq import ChatGroq
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.output_parsers import StrOutputParser
from tools import web_search, scrape_url
from dotenv import load_dotenv

load_dotenv()


# LLM
llm = ChatGroq(
    model="openai/gpt-oss-120b",
    temperature=0.7
)



def build_search_agent():
    return create_agent(
        model=llm,
        tools=[web_search],
        system_prompt=(
            "You are a helpful research assistant that can search the web "
            "for information. Use the web_search tool to find relevant "
            "information based on the user's query. "
            "Provide concise and accurate responses."
        )
    )



def build_reader_agent():
    return create_agent(
        model=llm,
        tools=[scrape_url],
        system_prompt=(
            "You are a helpful research assistant that can read web pages. "
            "Use the scrape_url tool to fetch and extract useful information "
            "from provided URLs. Provide clear and concise summaries."
        )
    )



writer_prompt = ChatPromptTemplate.from_messages([
    (
        "system",
        """You are a helpful assistant that writes well-structured,
factual and informative articles based on research provided to you.

Use only the information provided in the research.
Do not invent facts or sources."""
    ),
    (
        "user",
        """Write a detailed article on the topic below.

Topic:
{topic}

Research Gathered:
{research}

Structure the report as:

1. Introduction
2. Key Findings
   - Include at least 3 well-explained points
3. Conclusion
4. Sources
   - List all URLs found in the research

Be detailed, factual, informative and well-structured."""
    )
])


writer_chain = writer_prompt | llm | StrOutputParser()

critic_prompt = ChatPromptTemplate.from_messages([
    ("system", "You are a helpful assistant that critiques articles for clarity, structure, and factual accuracy."),
    ("user", """Critique the following article Provide feedback on clarity. Report:
        {report}
    
        Respond in this exact format:
        SCORE: X/10 
        Strengths: 
        - ... 
        - ...
        Areas to Imporove:
        - ...
        -...
        One line verdict:""")    
])

critic_chain = critic_prompt | llm | StrOutputParser()