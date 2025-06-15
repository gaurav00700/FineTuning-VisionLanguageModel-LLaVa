import os
import logging
import wandb
from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())  # load secrets from .env file

from unsloth import FastLanguageModel, FastVisionModel, is_bfloat16_supported, PatchDPOTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
import torch

# Configurations
max_seq_length = 4096 # Choose any! We auto support RoPE Scaling internally!
load_in_4bit = True # Use 4bit quantization to reduce memory usage. Can be False.
# If a model name ends with just bnb-4bit, without "unsloth", it refers to a standard BitsAndBytes 4-bit quantization.
BASE_MODEL = "llava-hf/llava-v1.6-mistral-7b-hf" #"unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit", "llava-hf/llava-v1.6-mistral-7b-hf", "unsloth/Llama-3.2-11B-Vision-Instruct",
DATASET_NAME = "openbmb/RLAIF-V-Dataset"    # HF ORPO dataset must have columns ["prompt", "image(PIL)", "question" "chosen", "rejected"]
OUTPUT_DIR = "outputs"  # for saving the logs
RUN_NAME = "llava-v1.6-mistral-7b"  # for tracking and logging

# Wandb login
wandb_token = os.environ.get('wandb_token')
wandb.login(key=wandb_token)
os.environ["WANDB_PROJECT"] = "FineTuning-llava-v1.6-mistral-7b"
os.environ["WANDB_LOG_MODEL"] = RUN_NAME

# Load model and tokenizer from Base LLM
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = BASE_MODEL, 
    dtype= None, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    use_exact_model_name = False, # Use the exact model to prevent 4bit
    load_in_4bit = load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
    load_in_8bit = False,
    full_finetuning = False,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)

# Use AutoTokenizer to get 1D output instead of 2D from FastVisionModel of Unsloth
auto_tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL, trust_remote_code=True)

# Add LoRA adapters to Base model so we only need to update 1 to 10% of all parameters!
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers     = True, # False if not finetuning vision layers
    finetune_language_layers   = True, # False if not finetuning language layers
    finetune_attention_modules = True, # False if not finetuning attention layers
    finetune_mlp_modules       = True, # False if not finetuning MLP layers
    r = 8,  # The larger, the higher the accuracy, but might overfit. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = 16,    # Recommended alpha == r at least. scales the weights of the adapters (more influence on base model), 16 was recommended on reddit
    lora_dropout = 0,   # Supports any, but = 0 is optimized
    bias = "none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,
    use_rslora = False,  # We support rank stabilized LoRA. scales lora_alpha with 1/sqrt(r), huggingface says this works better
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
    compile_model=False # Prevent torch.compile error with Mistral (or similar models) LM. MistralRotaryEmbedding.forward -> torch.compile -> generator
)

# casts the model to float32. Some layers of the lora adapter is in float16 causing runtime error during forward pass
# model = model.to(torch.float32)

# Load the Dataset
dataset = load_dataset(
    path = DATASET_NAME,
    split = "train[:5000]"   # split="train[:5000]"
    )  

# Split in Train and test dict
dataset_dict = dataset.train_test_split(test_size=0.10)


# Preprocess the loaded dataset
# The data must be formatted with appropriate prompt template first.
# See details here: https://github.com/huggingface/trl/blob/main/examples/scripts/orpo.py
# Reference : https://github.com/unslothai/unsloth/issues/2214

# Prompt template
alpaca_prompt = """You are a helpful AI assistant that answer the question based on accompanying image.
### Question:
{question}
### Response:
{response}"""

# End of sentence token
EOS_TOKEN = auto_tokenizer.eos_token # Must add EOS_TOKEN from tokenizer or auto_tokenizer
if EOS_TOKEN is None: EOS_TOKEN = "<eos>"

def format_prompt(sample):
    """Formats a sample for ORPO training.
    Args:
        sample (dict): A dictionary containing 'question', 'image', 'chosen', and 'rejected' keys.
                       'image' should contain a textual description of the image.
    Returns:
        dict: A dictionary formatted for ORPO training with 'image', 'prompt', 'chosen', and 'rejected' keys.
    """
    # instruction = sample["instruction"]
    question = sample["question"]
    image = sample["image"] # Usually an image path or PIL.Image
    chosen = sample["chosen"]
    rejected = sample["rejected"]

    # ORPOTrainer expects prompt/chosen/rejected keys
    # See: https://huggingface.co/docs/trl/main/en/orpo_trainer
    # sample["prompt"]   = alpaca_prompt.format(prompt=prompt, image=image, response="")
    sample["prompt"]   = alpaca_prompt.format(question=question, response="")
    sample["chosen"]   = chosen + EOS_TOKEN
    sample["rejected"] = rejected + EOS_TOKEN

    return sample

# Format training and testing dataset
dataset_train = dataset_dict['train'].map(
    format_prompt, 
    # fn_kwargs = {"tokenizer": tokenizer, "task": "ORPO"},
    remove_columns=['ds_name', 'origin_dataset', 'origin_split', 'idx', 'image_path'], # [],dataset.column_names. ["input_ids", "chosen_response_ids"]
    num_proc=6,  # Number of Processes
    # desc = "formatting train split"
    )

dataset_test = dataset_dict['test'].map(
    format_prompt, 
    # fn_kwargs = {"tokenizer": tokenizer, "task": "ORPO"},
    remove_columns=['ds_name', 'origin_dataset', 'origin_split', 'idx', 'image_path'], # [],dataset.column_names. ["input_ids", "chosen_response_ids"]
    num_proc=6,  # Number of Processes
    # desc = "formatting test split"
    )

# Train/Finetune the model
# Enable reward modelling stats
PatchDPOTrainer()

# Get the model trainer
orpo_trainer = ORPOTrainer(
    model = model,
    train_dataset = dataset_train,
    eval_dataset = dataset_test,
    # tokenizer = tokenizer,    # Not working, bug in unsloth
    processing_class = auto_tokenizer,
    evaluation_steps = 10,  # speed up evaluation: reduce the evaluation dataset size or change evaluation_steps
    args = ORPOConfig(
        # warmup_ratio = 0.1,
        # warmup_steps = 10, # overrides the warmup_ratio
        # learning_rate = 1e-6, # Lower for slower but more precise fine-tuning. Try values like 1e-4, 5e-5, or 2e-5
        max_length = max_seq_length,
        max_prompt_length = max_seq_length//2,
        max_completion_length = max_seq_length//2,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Simulates a larger batch size without increasing memory usage.
        beta = 0.1,
        logging_steps = 1,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        # num_train_epochs = 1 # full training runs (complete data is seen). Recommended [1-3], other overfitting
        max_steps = 50, # times weights are updated (good for large dataset)
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        output_dir = OUTPUT_DIR,
        report_to = "wandb", # Use this for WandB etc
        save_steps = 50,
        run_name = RUN_NAME
        # dataset_num_proc = 1 # Number of processes
    ),
)

# Start the training/finetuning
orpo_trainer.train()

# Evaluating the model performance
orpo_trainer.evaluate()

# Saving, loading finetuned models
LoRA_MODEL = f"{BASE_MODEL}_LoRA"
model.save_pretrained(LoRA_MODEL)  # Local saving
tokenizer.save_pretrained(LoRA_MODEL)