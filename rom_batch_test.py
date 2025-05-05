import asyncio
from agent_module import Agent, GPTModel, GeminiModel, InMemoryHistoryManager, load_env_file
from agent_module import get_prompt, get_sentences, save_as_md
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
o3-mini-2025-01-31
gpt-4.1-2025-04-14
gpt-4o-2024-08-06
'''

INPUT_FILE_PATH = os.path.join(os.getcwd(), "prompt", "input.txt")
SENTENCES = get_sentences(INPUT_FILE_PATH)
OUTPUT_DIR = os.path.join(os.getcwd(), "prompt")

GENERATION_MODEL_NAME = "o4-mini-2025-04-16"
VERIFICATION_MODEL_NAME = "gemini-2.5-pro-exp-03-25"

GENERATION_AGENT = Agent(
    name="Generation Agent",
    llm_interface=GPTModel(
        model=GENERATION_MODEL_NAME ,
        temperature=1,
    ),
    system_prompt=GENERATION_PROMPT
)

VERIFICATION_AGENT = Agent(
    name="Verification Agent",
    llm_interface=GeminiModel(
        model=VERIFICATION_MODEL_NAME,
        temperature=0.7
    ),
    system_prompt=VERIFICATION_PROMPT
)


async def rom_generation_batch_test(sentences, verify):
    # Import tqdm for progress bar
    from tqdm import tqdm

    def create_md_block(sent, rom):
        title = f"### {sent}"
        content = f"```\n{rom}\n```"
        return f"{title}\n{content}"

    results = []
    # Add tqdm wrapper around the for loop
    for sentence in tqdm(sentences, desc="Processing sentences"):
        result = await GENERATION_AGENT.process(sentence)
        if verify:
            result = await VERIFICATION_AGENT.process(f"Original sentence: {sentence}\nGenerated Relations: \n{result}")
        results.append(create_md_block(sentence, result))

    filename = f'GEN-{GENERATION_AGENT.model_name()}'
    if verify:
        filename += f'-VER-{VERIFICATION_AGENT.model_name()}'
    output_path = os.path.join(OUTPUT_DIR, filename)
    md_content = "\n---\n".join(results)
    save_as_md(md_content, output_path)

asyncio.run(rom_generation_batch_test(SENTENCES, False))
