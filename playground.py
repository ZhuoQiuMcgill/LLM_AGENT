import asyncio
from agent_module import Agent, GPTModel, InMemoryHistoryManager, load_env_file

# Load environment variables from .env file
load_env_file()

# Create an LLM implementation
llm = GPTModel(model="gpt-4o-mini-2024-07-18")

# Create a history manager (optional, Agent will create one if not provided)
history = InMemoryHistoryManager(max_messages=50)

# Create an agent
agent = Agent(
    llm_interface=llm,
    history_manager=history,
    name="MyAssistant",
    system_prompt="You are a helpful assistant specialized in Python programming."
)


# Use the agent
async def chat():
    response = await agent.process("Hello! Can you help me with a Python problem?")
    print(response)

    # Continue the conversation
    response = await agent.process("How do I create a list comprehension?")
    print(response)


# Run the async function
asyncio.run(chat())
