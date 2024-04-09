import os
from dotenv import load_dotenv
from langchain_openai import ChatOpenAI
from langchain_community.tools import DuckDuckGoSearchRun
from crewai import Agent, Task, Crew, Process
from crewai_tools import SerperDevTool

# load environment variables that specify the llm(s) to use, API keys and API endpoints.
load_dotenv()

# define a llm based on a hosted ollama model. This is not required when using ChatGPT.
# we'll use this as an optional parameter when instantiating each agent.
llm = ChatOpenAI(
    openai_api_key=os.environ["OPENAI_API_KEY"],
    model=os.environ["OPENAI_MODEL_NAME"],
    base_url=os.environ["OPENAI_API_BASE"],
)

# SerperDevTool provides an API interface to Google search. This is the only tool we'll
# provide to our AI agents.
# search_tool = SerperDevTool()

# Alternatively we can use DuckDuckGoSearch
search_tool = DuckDuckGoSearchRun()

# Create a senior researcher agent to explore and share knowledge on a given topic. Enabling memory
# (to remember past interactions) and verbose mode to assist with debugging and understand what it's doing.
researcher = Agent(
    role="Senior Researcher",
    goal="Uncover groundbreaking technologies in {topic}",
    verbose=True,
    memory=True,
    max_rpm=12,
    max_iter=12,
    backstory=(
        "You are a world renowned senior science and technology researcher"
        "You search the internet for articles on {topic} future trends"
        "and provide sub topics and URLs to writers to base articles on"
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=False,
)

# Create a writer agent to refine and simplify research material. Enabling memory (to remember past interactions)
# and verbose mode to assist with debugging and understand what it's doing.
writer = Agent(
    role="Senior Writer",
    goal="Write a compelling and interesting science and technology article based on {topic}",
    verbose=True,
    memory=True,
    max_rpm=12,
    max_iter=12,
    backstory=(
        "You are a world renowned senior science and technology author. "
        "You work from briefing documents that guide your writing focus. "
        "You search the internet for articles on {topic} to support the outline in the briefing document. "
        "You develop a clear, compelling narrative that's accessible to a layperson reader."
        "You always include references to all source material used in articles."
    ),
    tools=[search_tool],
    llm=llm,
    allow_delegation=False,
)

# Create a new task to perform the research. We associate the task with the researcher agent
# and restrict its access to only the search_tool.
research_task = Task(
    description=(
        "Identify future trends around {topic}. "
        "Limit the number of trends to at most three areas. "
        "Focus on identifying the pros and cons of each trends, the market opportunity and risks. "
        "Create a briefing document for a writer to continue your research. "
        "Clearly articulate the key points and the narrative you want in the final article. "
    ),
    expected_output="A three to five paragraph briefing document on the latest trends around {topic} formatted in markdown.",
    tools=[search_tool],
    verbose=True,
    memory=True,
    max_rpm=12,
    max_iter=12,
    agent=researcher,
    async_execution=False,
    output_file="research-notes.md",
)

# Create a new task to perform the writing. We associate the task with the writer agent
# and restrict its access to only the search_tool. We specify it should output the article
# in markdown format and store it in file article-draft.md
write_task = Task(
    description="""Using the insights provided, write an engaging blog post
    article that highlights the most significant future advancements.
    Your post should be informative yet accessible, catering to a techonolgy audience.
    Avoid complex words and phrases.""",
    # expected_output="Full blog post of at least 4 paragraphs",
    expected_output="A five to seven paragraph article that covers the latest trends around {topic} formatted in markdown.",
    tools=[search_tool],
    verbose=True,
    memory=True,
    max_rpm=12,
    max_iter=12,
    agent=writer,
    output_file="article-draft.md",  # Example of output customization
)

# We now define a crew consisting of two agents - researcher and writer - and provide it with an
# ordered list of tasks to complete sequentially.
crew = Crew(
    agents=[researcher, writer],
    tasks=[research_task, write_task],
    process=Process.sequential,  # Optional: Sequential task execution is default
    memory=True,
    verbose=9,
    cache=True,
    max_rpm=12,
    max_iter=24,
    share_crew=False,
)

# Ask the user for an input topic and then put the crew to work.
result = crew.kickoff(
    inputs={"topic": input("Please provide a research topic: ").strip()}
)
print(result)
