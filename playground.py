import asyncio
from agent_module import Agent, GPTModel, InMemoryHistoryManager, load_env_file
import os

# Load environment variables from .env file
load_env_file()


def get_prompt(filepath):
    try:
        with open(filepath, 'r', encoding='utf-8') as file:
            content = file.read()
        return content
    except Exception as e:
        return f"Error reading file: {e}"


ROM_PROMPT_PATH = os.path.join(os.getcwd(), "prompt", "ROM.md")
SYSTEM_INSTRUCTION = get_prompt(ROM_PROMPT_PATH)

# Create an LLM implementation
llm = GPTModel(model="o4-mini-2025-04-16")

# Create a history manager (optional, Agent will create one if not provided)
history = InMemoryHistoryManager(max_messages=50)

# Create an agent
agent = Agent(
    llm_interface=llm,
    history_manager=history,
    name="ROM Generator",
    system_prompt=SYSTEM_INSTRUCTION
)


# Use the agent
async def chat():
    '''
    # First test the connection to verify API keys and model availability
    print("Testing connection to LLM...")
    try:
        test_response = await agent.connection_test()
        print(f"Connection test result: {test_response}")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return  # Exit if connection test fails
    '''

    # Now proceed with normal conversation if connection test passed
    print("\nStarting conversation...")
    prompt = "She smiled as she read about the time they built a treehouse together."
    print(f'User:\n{prompt}\n')

    response = await agent.process(prompt)
    print(f'{agent.name}:\n{response}\n')


# Run the async function
asyncio.run(chat())
