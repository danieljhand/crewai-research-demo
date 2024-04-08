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
#search_tool = SerperDevTool(n_results=3)

# Alternatively we can use DuckDuckGoSearch
search_tool = DuckDuckGoSearchRun()

# Create a senior researcher agent to explore and share knowledge on a given topic. Enabling memory 
# (to remember past interactions) and verbose mode to assist with debugging and understand what it's doing.
researcher = Agent(
  role='Senior Researcher',
  goal='Uncover groundbreaking technologies in {topic}',
  verbose=True,
  memory=True,
  max_rpm=12, # No limit on requests per minute
  max_iter=12,
  backstory=(
    "You are a world renowned senior science and technology researcher"
    "You search the internet for articles on {topic} future trends"
    "and provide sub topics and URLs to writers to base articles on"
  ),
  tools=[search_tool],
  llm=llm,
  allow_delegation=False
)

editor = Agent(
  role='Senior Editor',
  goal='Review and provide feedback on how to improve an article written on {topic}',
  verbose=True,
  memory=True,
  backstory=(
      "You are an editor for a prestigious academic journal."
  ),
  tools=[],
  llm=llm,
  allow_delegation=False
)

# Create a writer agent to refine and simplify research material. Enabling memory (to remember past interactions)
# and verbose mode to assist with debugging and understand what it's doing.
writer = Agent(
  role='Writer',
  goal='Narrate compelling tech stories around {topic}',
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
    "Identify future trends around {topic}. "
    "Limit the number of trends to at most three areas. "
    "Focus on identifying the pros and cons of each trends, the market opportunity and risks. "
    "Create a briefing document for a writer to continue your research. "
    "Clearly articulate the key points and the narrative you want in the final report. "
  ),
  expected_output="A three to five paragraph briefing document on the latest trends around {topic} formatted in markdown ",
  tools=[search_tool],
  agent=researcher,
  async_execution=False,
  output_file='research-notes.md'
)


# Create a new task to review the report and provide clear guidance for improvement.
review_task = Task(
  description=(
      "A new research paper has been submitted that explores future developments in {topic}. Your role is to analyze the paper and provide feedback to the author(s).  What steps do you take to ensure the research is sound, the writing is clear and concise, and the paper is suitable for publication in your journal?"
  ),
  expected_output="A comprehensive 5 paragraph report highlighting the main areas of improvement and suggestions on how to improve",
  tools=[],
  agent=editor,
  async_execution=False,
  output_file='editor-note.txt'
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
  # agents=[researcher, writer, editor],
  # tasks=[research_task, write_task, review_task, write_task],
  agents=[researcher],
  tasks=[research_task],
  process=Process.sequential,  # Optional: Sequential task execution is default
  memory=True,
  verbose=1,
  cache=True,
  max_rpm=10,
  share_crew=False
)


# Ask the user for an input topic and then put the crew to work.

result = crew.kickoff(inputs={'topic': input("Please provide a research topic: ").strip()})
print(result)
