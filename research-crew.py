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
    model = os.environ["OPENAI_MODEL_NAME"],
    base_url = os.environ["OPENAI_API_BASE"]
)

# SerperDevTool provides an API interface to Google search. This is the only tool we'll
# provide to our AI agents.
search_tool = SerperDevTool()

# Alternatively we can use DuckDuckGoSearch
# search_tool = DuckDuckGoSearchRun()

# Create a senior researcher agent to explore and share knowledge on a given topic. Enabling memory 
# (to remember past interactions) and verbose mode to assist with debugging and understand what it's doing.
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "You are a world renowned senior science and technology researcher respected for the quality and reliability of your research."
    "You are also an excellent collaborator often working with a team of writers that create engaging and informative articles based on your research."
  ),
  tools=[search_tool],
  llm=llm,
  allow_delegation=False
)

# Create a writer agent to refine and simplify research material. Enabling memory (to remember past interactions)
# and verbose mode to assist with debugging and understand what it's doing.
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories about {topic}',
  verbose=True,
  memory=True,
  backstory=(
    "With a flair for simplifying complex topics, you craft"
    "engaging narratives that captivate and educate, bringing new"
    "discoveries to light in an accessible manner."
  ),
  tools=[search_tool],
  llm=llm,
  allow_delegation=False
)

# Create a new task to perform the research. We associate the task with the researcher agent
# and restrict its access to only the search_tool.
research_task = Task(
  description=(
    "Identify the next big trend in {topic}."
    "Focus on identifying pros and cons and the overall narrative."
    "Your final report should clearly articulate the key points"
    "its market opportunities and potential risks."
  ),
  expected_output="A comprehensive 3 paragraph report on the latest {topic} trends.",
  tools=[search_tool],
  agent=researcher,
  async_execution=False,
  output_file='research-note.txt'
)

# Create a new task to perform the writing. We associate the task with the writer agent
# and restrict its access to only the search_tool. We specify it should output the article
# in markdown format and store it in file new-blog-post.md
write_task = Task(
  description=(
    "Compose an insightful article on {topic}."
    "Focus on the latest trends and how it's impacting the industry."
    "This article should be easy to understand, engaging and positive."
  ),
  expected_output="A 4 paragraph article on {topic} advancements formatted as markdown.",
  tools=[search_tool],
  agent=writer,
  async_execution=False,
  output_file='new-blog-post.md'  # Example of output customization
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
  max_rpm=10,
  share_crew=False
)


# Ask the user for an input topic and then put the crew to work.

result = crew.kickoff(inputs={'topic': input("Please provide a research topic: ").strip()})
print(result)
