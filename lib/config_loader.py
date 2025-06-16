import os
import yaml
from dotenv import load_dotenv, find_dotenv
from pydantic import BaseModel, Field, root_validator, model_validator
from typing import Optional

class Config(BaseModel):
    # === Model and Dataset ===
    base_model: str = "unsloth/llava-v1.6-mistral-7b-hf-bnb-4bit"
    dataset_name: str = "openbmb/RLAIF-V-Dataset"

    # === Training Options ===
    evaluation: bool = True
    max_seq_length: int = 4096
    load_in_4bit: bool = True
    load_in_8bit: bool = False
    full_finetuning: bool = False
    dtype: Optional[str] = None  # e.g., "float16", "bfloat16", or None
    num_train_epochs: Optional[int] = None
    max_steps: int = 100

    # === Output and Logging ===
    output_dir: str = "outputs"
    run_name: str
    wandb_project: str
    wandb_api_key: Optional[str] = None

    # === LoRA Config ===
    finetune_vision_layers: bool = True, # False if not finetuning vision layers
    finetune_language_layers: bool = True, # False if not finetuning language layers
    finetune_attention_modules:bool = True, # False if not finetuning attention layers
    finetune_mlp_modules: bool = True, # False if not finetuning MLP layers
    r: int = 16     # The larger, the higher the accuracy, but might overfit. Choose any number > 0 ! Suggested 8, 16, 32, 64, 128
    lora_alpha: int = 16    # LoRA Rank * 1 or 2. scales the weights of the adapters (more influence on base model)
    lora_dropout: float = 0.0
    bias: str = "none"
    use_rslora: bool = False
    compile_model: bool = False

    # === Computed Fields ===
    save_model_path: Optional[str] = None

    @root_validator(pre=True)
    def fill_computed(cls, values):
        output_dir = values.get("output_dir", "outputs")
        run_name = values.get("run_name")
        if run_name:
            values["save_model_path"] = f"{output_dir}/LoRA/{run_name}"
        return values

def load_config(config_path: str) -> Config:

    # Load .env if present
    load_dotenv(find_dotenv())

    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Config file not found: {config_path}")

    # Load config
    with open(config_path, "r") as f:
        data = yaml.safe_load(f)

    # Parse and validate config
    config = Config(**data)

    # Handle wandb token from .env or config
    # wandb_api_key = config.wandb_api_key or os.getenv("wandb_api_key")
    if config.wandb_api_key != 'YOUR_WANDB_TOKEN':
        wandb_api_key = config.wandb_api_key
    else:
        wandb_api_key = os.getenv("wandb_api_key")

    if not wandb_api_key:
        print("[INFO] WandB 'wandb_api_key' missing: provide it in config.yaml or .env")
        print("[INFO] Logging is not active due to missing 'wandb_api_key")
    
    # Set env vars
    os.environ["WANDB_API_KEY"] = wandb_api_key
    os.environ["WANDB_PROJECT"] = config.wandb_project
    os.environ["WANDB_LOG_MODEL"] = config.run_name

    return config

