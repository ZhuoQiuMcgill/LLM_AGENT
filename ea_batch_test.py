import asyncio
from agent_module import Agent, GPTModel, GeminiModel, InMemoryHistoryManager, load_env_file
from agent_module import get_prompt, get_sentences, save_as_md, read_json_to_dict
import os
# Import tqdm for progress bar
from tqdm import tqdm

# Load environment variables from .env file
load_env_file()

STEP1_PROMPT_PATH = os.path.join(os.getcwd(), "prompt", "ea_prompts", "ea_prompt_step1.md")
STEP2_PROMPT_PATH = os.path.join(os.getcwd(), "prompt", "ea_prompts", "ea_prompt_step2.md")

SYSTEM_INSTRUCTION_STEP1 = get_prompt(STEP1_PROMPT_PATH)
SYSTEM_INSTRUCTION_STEP2 = get_prompt(STEP2_PROMPT_PATH)
INPUT_PATH = os.path.join(os.getcwd(), "prompt", "ea_input.json")
INPUT_LIST = read_json_to_dict(INPUT_PATH)["input_list"][:1]

'''
OPEN AI MODEL LIST:
o4-mini-2025-04-16
o3-mini-2025-01-31
gpt-4.1-2025-04-14
gpt-4o-2024-08-06
'''

INPUT_FILE_PATH = os.path.join(os.getcwd(), "prompt", "input.txt")
SENTENCES = get_sentences(INPUT_FILE_PATH)
OUTPUT_DIR = os.path.join(os.getcwd(), "prompt", "ea_outputs")

OPENAI_MODEL_NAME = "o4-mini-2025-04-16"
GOOGLE_MODEL_NAME = "gemini-2.5-pro-exp-03-25"

GPT_STEP1 = Agent(
    name="Generation Agent",
    llm_interface=GPTModel(
        model=OPENAI_MODEL_NAME,
        temperature=1,
    ),
    system_prompt=STEP1_PROMPT_PATH
)

GPT_STEP2 = Agent(
    name="Generation Agent",
    llm_interface=GPTModel(
        model=OPENAI_MODEL_NAME,
        temperature=1,
    ),
    system_prompt=STEP2_PROMPT_PATH
)

GEMINI_STEP1 = Agent(
    name="Verification Agent",
    llm_interface=GeminiModel(
        model=GOOGLE_MODEL_NAME,
        temperature=0
    ),
    system_prompt=STEP1_PROMPT_PATH
)

GEMINI_STEP2 = Agent(
    name="Verification Agent",
    llm_interface=GeminiModel(
        model=GOOGLE_MODEL_NAME,
        temperature=0
    ),
    system_prompt=STEP2_PROMPT_PATH
)

GEMINI_AGENTS = [GEMINI_STEP1]
GPT_AGENTS = [GPT_STEP1, GPT_STEP2]


async def steps_ea_batch_test(inputs, agents):
    for i, input_text in enumerate(tqdm(inputs, desc="Processing sentences")):
        result = input_text
        for j, agent in enumerate(agents):
            print(f'Prompt: {result}')
            response = await agent.process(result)
            result = f'{result}\n---\n### Step{j + 1}: \n{response}'
            print(f'step{j + 1} completed!')
            print(result)
        filename = f"MultiSteps-{agents[0].model_name()}"
        save_as_md(result, os.path.join(OUTPUT_DIR, filename))


async def ea_generation_batch_test(inputs, agent):
    # Add tqdm wrapper around the for loop
    for i, input_text in enumerate(tqdm(inputs, desc="Processing sentences")):
        result = await agent.process(input_text)
        filename = f'EA-{agent.model_name()}-{i}'
        save_as_md(result, os.path.join(OUTPUT_DIR, filename))


# asyncio.run(ea_generation_batch_test(INPUT_LIST, GEMINI))
asyncio.run(steps_ea_batch_test(INPUT_LIST, GEMINI_AGENTS))

