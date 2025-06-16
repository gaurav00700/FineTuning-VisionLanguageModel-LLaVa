# Preprocess the loaded dataset
# The data must be formatted with appropriate prompt template first.
# See details here: https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py
# Reference : https://github.com/unslothai/unsloth/issues/2214

# Alpaca prompt template
alpaca_prompt = """You are a helpful AI assistant that answer the question based on accompanying image.
### Question:
{question}
### Response:
{response}"""

def format_fn(sample:dict, eos_token:str, prompt_template:str = alpaca_prompt) -> dict:
    """Format the dataset for training
    Args:
        - sample (dict): A dictionary containing 'question', 'image', 'chosen', and 'rejected' keys. 'image' should contain a textual description of the image.
        - prompt_template (str): Prompt template compilable to Base model
        - eos_token (str): End of sentence token. Usually from tokenizer
    Returns:
        - dict: A dictionary formatted for ORPO training with 'image', 'prompt', 'chosen', and 'rejected' keys. 
    """
    # instruction = sample["instruction"]
    question = sample["question"]
    image = sample["image"] # Usually an image path or PIL.Image
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    # ORPOTrainer expects prompt/chosen/rejected keys
    # See: https://huggingface.co/docs/trl/main/en/orpo_trainer
    # sample["prompt"]   = prompt_template.format(prompt=prompt, image=image, response="")
    sample["prompt"]   = prompt_template.format(question=question, response="")
    sample["chosen"]   = chosen + eos_token
    sample["rejected"] = rejected + eos_token

    return sample