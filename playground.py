import asyncio
from agent_module import Agent, GPTModel, GeminiModel, InMemoryHistoryManager, load_env_file
from agent_module import get_prompt, get_sentences
import os

# Load environment variables from .env file
load_env_file()

GENERATION_PROMPT_PATH = os.path.join(os.getcwd(), "prompt", "rom_generation_prompt.md")
GENERATION_PROMPT = get_prompt(GENERATION_PROMPT_PATH)
VERIFICATION_PROMPT_PATH = os.path.join(os.getcwd(), "prompt", "rom_verification_prompt.md")
VERIFICATION_PROMPT = get_prompt(VERIFICATION_PROMPT_PATH)

'''
OPEN AI MODEL LIST:
o4-mini-2025-04-16
gpt-4.1-2025-04-14
gpt-4o-2024-08-06
o3-mini-2025-01-31
'''

SENTENCE = "It was one of the happiest moments of her life."
# Create an LLM implementation
openai_llm = GPTModel(
    model="o3-mini-2025-01-31",
    temperature=1,
)

google_llm = GeminiModel(
    model="gemini-2.5-pro-exp-03-25",
    temperature=0
)

# Create a history manager (optional, Agent will create one if not provided)
history = InMemoryHistoryManager(max_messages=50)

# Create an agent
generation_agent = Agent(
    llm_interface=openai_llm,
    name="ROM Generation",
    system_prompt=GENERATION_PROMPT
)

verification_agent = Agent(
    llm_interface=google_llm,
    name="ROM Verification",
    system_prompt=VERIFICATION_PROMPT
)


async def connection_test():
    print("Testing connection to LLM...")
    try:
        test_response = await generation_agent.connection_test()
        print(f"Generation Step Connection test result: {test_response}")
        test_response = await verification_agent.connection_test()
        print(f"Verification Step Connection test result: {test_response}")
    except Exception as e:
        print(f"Connection test failed: {str(e)}")
        return  # Exit if connection test fails


# Use the agent
async def chat():
    # await connection_test()
    # print("\nStarting conversation...")
    print(f'Original sentence: \n{SENTENCE}\n')

    response = await generation_agent.process(SENTENCE)
    print(f'Generation step by {generation_agent.model_name()}: \n{response}\n')

    response = await verification_agent.process(f"Original sentence: {SENTENCE}\nGenerated Relations: \n{response}")
    print(f'Verification step by {verification_agent.model_name()}: \n{response}\n')


# Run the async function
asyncio.run(chat())
