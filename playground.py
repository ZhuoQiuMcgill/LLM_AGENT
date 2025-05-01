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
    name="GPT-4o-mini",
    system_prompt="You are a design expert"
)


# Use the agent
async def chat():
    # First test the connection to verify API keys and model availability
    print("Testing connection to LLM...")
    try:
        test_response = await agent.connection_test()
        print(f"Connection test result: {test_response}")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return  # Exit if connection test fails

    # Now proceed with normal conversation if connection test passed
    print("\nStarting conversation...")
    prompt = "Hello! Can you help me design problem?"
    print(f'User:\n {prompt}\n')

    response = await agent.process(prompt)
    print(f'{agent.name}:\n {response}\n')

    # Continue the conversation
    prompt = "Please design a house that can travel from one location to another location."
    print(f'User:\n {prompt}\n')
    response = await agent.process(prompt)
    print(f'{agent.name}:\n {response}\n')


# Run the async function
asyncio.run(chat())
