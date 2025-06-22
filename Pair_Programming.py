from dotenv import load_dotenv
import os
from crewai import Task,Crew, Agent, LLM
from langchain_google_genai import ChatGoogleGenerativeAI

load_dotenv()

llm = LLM(
    model="gemini/gemini-2.0-flash",
    temperature=0.7,
    api_key=os.getenv("GOOGLE_API_KEY")
)             

code_writer_agent = Agent(
    role="Software Engineer",
    goal="Write optimized code for a given task",
    backstory="""
    You are a software engineer who python write code for given task.
    The code should be optimized, and maintainable and include doc strings, comments, etc.
    """,
    llm=llm,
    verbose=True
)

code_writer_task =  Task(
    description="""
    Write the code to solve the 
    Problem: {problem}
    """,
    expected_output="Well Formatted code to solve the problem, include type hinting",
    agent=code_writer_agent
)

code_reviewer_agent = Agent(
    role="Senior software engineer",
    goal="Make sure the code written is optimized and maintainable",
    backstory="""You are a Senior software engineer who reviews the code written for given task.
    You should check the code readability, maintainability, and performance""",
    llm=llm,
    verbose=True
)

code_reviewer_task =  Task(
    description="""
    A software engineer has written this code for the given problem in the python programming language.
    Review the code critically and make any changes to the code if necessary.
    'Problem': {problem}
    """,
    expected_output="Well formatted code after the review",
    agent=code_reviewer_agent
)

crew = Crew(
    agents=[code_writer_agent,code_reviewer_agent],
    tasks=[code_writer_task,code_reviewer_task],
    verbose=True
)

results = crew.kickoff(inputs={"problem":"Create a game of tic-tac-toe"})

print(results)