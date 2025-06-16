import os
import sys
import logging
import wandb

from unsloth import FastVisionModel, is_bfloat16_supported, PatchDPOTrainer
from transformers import AutoTokenizer
from datasets import load_dataset
from trl import ORPOConfig, ORPOTrainer
import torch

from lib.config_loader import load_config
from lib import helper

# ---------------------------
# Load Configuration
# ---------------------------
# Check for number of script argument
if len(sys.argv) < 2:
    print("Usage: python train.py path/to/config.yaml")
    sys.exit(1)

# Load configuration
config_path = sys.argv[1]
config = load_config(config_path)

# ---------------------------
# Setup Logging
# ---------------------------
logger = logging.getLogger("VLM Finetuning")
logger.setLevel(logging.INFO)
formatter = logging.Formatter("%(asctime)s - %(levelname)s - %(message)s")

# Console handler
console_handler = logging.StreamHandler(sys.stdout)
console_handler.setLevel(logging.INFO)
console_handler.setFormatter(formatter)
logger.addHandler(console_handler)

# File handler
log_file_dir = os.path.join(config.output_dir, "logs")
os.makedirs(log_file_dir, exist_ok=True)
log_file_path = os.path.join(log_file_dir, f"{config.run_name}.log")

file_handler = logging.FileHandler(log_file_path)
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(formatter)
logger.addHandler(file_handler)

# set success level
logging.SUCCESS = 25  # between WARNING and INFO
logging.addLevelName(logging.SUCCESS, 'SUCCESS')
setattr(logger, 'success', lambda message, *args: logger._log(logging.SUCCESS, message, args))

# ---------------------------
# Wandb Login
# ---------------------------
wandb_login = wandb.login()   # key=os.getenv['WANDB_API_KEY']

# ---------------------------
# Check GPU Availability
# ---------------------------
if torch.cuda.is_available():
    logger.info("PyTorch is connected to GPU.")
    logger.info(f"GPU Device Name: {torch.cuda.get_device_name(0)}")
    logger.info(f"Number of GPUs available: {torch.cuda.device_count()}")
    logger.info(f"Current GPU: {torch.cuda.current_device()}")
else:
    logger.warning("PyTorch is not connected to GPU.")


# ---------------------------
# Models
# ---------------------------
# Load model and tokenizer from Base LLM
model, tokenizer = FastVisionModel.from_pretrained(
    model_name = config.base_model, 
    use_exact_model_name = True, # Use the exact model to prevent 4bit download
    dtype= config.dtype, # None for auto detection. Float16 for Tesla T4, V100, Bfloat16 for Ampere+
    load_in_4bit = config.load_in_4bit, # Use 4bit to reduce memory use. False for 16bit LoRA.
    load_in_8bit = config.load_in_8bit,
    full_finetuning = config.full_finetuning,
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for long context
)
logger.success("Loaded the Base model")

# Use AutoTokenizer to get 1D output instead of 2D from FastVisionModel of Unsloth
auto_tokenizer = AutoTokenizer.from_pretrained(config.base_model, trust_remote_code=True)

# Add LoRA adapters to Base model so we only need to update 1 to 10% of all parameters!
model = FastVisionModel.get_peft_model(
    model,
    finetune_vision_layers = config.finetune_vision_layers, # False if not finetuning vision layers
    finetune_language_layers = config.finetune_language_layers, # False if not finetuning language layers
    finetune_attention_modules = config.finetune_attention_modules, # False if not finetuning attention layers
    finetune_mlp_modules = config.finetune_mlp_modules, # False if not finetuning MLP layers

    r = config.r,  # The larger, the higher the accuracy, but might overfit. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha = config.lora_alpha,    # LoRA Rank * 1 or 2. scales the weights of the adapters (more influence on base model)
    lora_dropout = config.lora_dropout,   # Supports any, but = 0 is optimized. Dropout rate to prevent overfitting
    bias = "none",  # Supports any, but = "none" is optimized
    use_gradient_checkpointing = "unsloth", # True or "unsloth" for very long context
    random_state = 3407,    # Seed for reproducibility
    max_seq_length = config.max_seq_length, # Context length
    use_rslora = False,  # We support rank stabilized LoRA. scales lora_alpha with 1/sqrt(r), huggingface says this works better
    loftq_config = None, # And LoftQ
    # target_modules = "all-linear", # Optional now! Can specify a list if needed
    compile_model=False # Prevent torch.compile error with Mistral (or similar models) LM. MistralRotaryEmbedding.forward -> torch.compile -> generator
)

logger.success("LoRA adapter is applied to Base model")
logger.info(model)

# casts the model to float32. Some layers of the lora adapter is in float16 causing runtime error during forward pass
# model = model.to(torch.float32)

# ---------------------------
# Dataset
# ---------------------------
# Load the Dataset
dataset = load_dataset(
    path = config.dataset_name,
    split = "train[:5000]"   # First 5000
    )  
logger.success("Dataset loaded")

# Split in train and test dict
dataset_dict = dataset.train_test_split(test_size=0.10)

# End of sentence token from tokenizer or auto_tokenizer
EOS_TOKEN = auto_tokenizer.eos_token 
if EOS_TOKEN is None: EOS_TOKEN = "<eos>"

# Preprocess the loaded dataset
logger.info("Processing Dataset ...")

# Format training dataset
dataset_train = dataset_dict['train'].map(
        helper.format_fn, 
        fn_kwargs = {"eos_token": EOS_TOKEN},
        remove_columns=['ds_name', 'origin_dataset', 'origin_split', 'idx', 'image_path'], # [],dataset.column_names. ["input_ids", "chosen_response_ids"]
        # num_proc=1,  # Number of Processes for parallelization
    )

# Format evaluation data
if config.evaluation:
    dataset_test = dataset_dict['test'].map(
        helper.format_fn, 
        fn_kwargs = {"eos_token": EOS_TOKEN},
        remove_columns=['ds_name', 'origin_dataset', 'origin_split', 'idx', 'image_path'], # [],dataset.column_names. ["input_ids", "chosen_response_ids"]
        # num_proc=1,  # Number of Processes for parallelization
    )

logger.success("Dataset processing finished")

# ---------------------------
# Training/Finetuning Setup
# ---------------------------
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
        max_length = config.max_seq_length,
        max_prompt_length = config.max_seq_length//2,
        max_completion_length = config.max_seq_length//2,
        per_device_train_batch_size = 1,
        gradient_accumulation_steps = 4, # Simulates a larger batch size without increasing memory usage.
        beta = 0.1,
        logging_steps = 1,
        optim = "adamw_8bit",
        lr_scheduler_type = "linear",
        # num_train_epochs = config.num_train_epochs, # full training runs (complete data is seen). Recommended [1-3], other overfitting
        max_steps = config.max_steps, # times weights are updated (good for large dataset)
        fp16 = not is_bfloat16_supported(),
        bf16 = is_bfloat16_supported(),
        output_dir = f"{config.output_dir}/checkpoints/{config.run_name}",
        report_to = "wandb" if wandb_login else "none", # Use this for WandB etc
        save_steps = 50,
        run_name = config.run_name
    ),
)

logger.success("Loaded ORPO trainer module")

# Start the training/finetuning
logger.info("Starting the model Training ....")
orpo_trainer.train(
    # resume_from_checkpoint='outputs/checkpoints/llava-hf_llava-v1.6-mistral-7b_8bit_r16/checkpoint-700'
    )
logger.success("Finished the model training")


# ---------------------------
# Evaluation
# ---------------------------
# Evaluating the model performance
if config.evaluation:
    logger.info("Starting Evaluation of Finetuned model ...")
    orpo_trainer.evaluate()
    logger.success("Finished the Evaluation")


# ---------------------------
# Save Model
# ---------------------------
# Saving, loading finetuned models
logger.info("Saving the finetuned Model/LoRA adapter ...")
# Local saving
logger.info("Saving Locally")
model.save_pretrained(config.save_model_path)  
tokenizer.save_pretrained(config.save_model_path)

# Online saving in HF repo
# logger.info("Saving in HF repo")
# model.push_to_hub("your_name/lora_model", token = "...") 
# tokenizer.push_to_hub("your_name/lora_model", token = "...")

logger.success("Script execution complete !!!")