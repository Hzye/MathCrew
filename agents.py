from crewai import Agent
from textwrap import dedent
from langchain.llms import OpenAI, Ollama
from langchain_openai import ChatOpenAI


"""
team:
- high school math subject coordinator
- high school math teacher
- math olympiad

agents must be result driven + have a clear goal in mind
role == job title
goal should be actionable
backstory == resume
"""

class CustomAgents:
    def __init__(self):
        self.OpenAIGPT35 = ChatOpenAI(model_name="gpt-3.5-turbo", temperature=0.7)
        self.OpenAIGPT4 = ChatOpenAI(model_name="gpt-4", temperature=0.7)
        self.Ollama = Ollama(model="openhermes")

    def math_coordinator_agent(self):
        return Agent(
            role="High School Math Subject Co-ordinator",
            backstory=dedent(f"""
                Expert in aligning math content with curriculum. 
                I have decades of experience running the math department at a top high school. 
            """),
            goal=dedent(f"""
                Create relevant math homework worksheet or practice exam based on provided topics and requirements.
            """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def math_teacher_agent(self):
        return Agent(
            role="High School Math Teacher",
            backstory=dedent(f"""
                Expert in teaching math and writing math questions.
                I have decades of experience as a math teacher at a top high school.
            """),
            goal=dedent(f"""
                Collect the most relevant math questions based on provided topics and requirements.
                Can also adjust problems.
            """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )

    def math_olympiad_agent(self):
        return Agent(
            role="Math Olympiad Winner",
            backstory=dedent(f"""
                Expert in solving and stepping through math problems.
                I am a multi-year Math Olympiad winner.
            """),
            goal=dedent(f"""
                Solve and explain the solutions to all math problems.
            """),
            # tools=[tool_1, tool_2],
            allow_delegation=False,
            verbose=True,
            llm=self.OpenAIGPT35,
        )
