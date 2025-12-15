#!/usr/bin/env python3
"""
models processing class using the transformers library
Note that the inference here is not powered by vllm
"""

# package imports
from transformers import AutoModelForCausalLM, AutoTokenizer
from dotenv import load_dotenv
import click # for CLI
import os

load_dotenv()

# default huggingface cache
# HF_HOME should point to the parent directory (e.g., ~/.cache/huggingface)
# transformers will then use HF_HOME/hub for the actual cache
default_hf_home = "~/.cache/huggingface"
default_hf_cache = os.path.join(os.path.expanduser(default_hf_home), "hub")

# Set HuggingFace cache directory to use local cache (only if not already set)
if "HF_HOME" not in os.environ:
    os.environ["HF_HOME"] = os.path.expanduser(default_hf_home)

# Get the cache directory - this is where the actual model files are stored
# (HF_HOME/hub, not just HF_HOME)
HF_CACHE_DIR = os.path.join(
    os.environ.get("HF_HOME", os.path.expanduser(default_hf_home)),
    "hub"
)

class ModelHandler:
    def __init__(self, model_name: str, local_files_only: bool = False):
        # initialise ModelHandler with model loaded
        self.local_files_only = local_files_only
        self.model_name = model_name
        self.load_model()

    def load_model(self):
        # loading model
        if self.check_model_in_cache(): 
            # check whether the model is downloaded in the local cache
            # if it is, load this model
            self.model = AutoModelForCausalLM.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                cache_dir=HF_CACHE_DIR
            )
            self.tokenizer = AutoTokenizer.from_pretrained(
                self.model_name,
                local_files_only=self.local_files_only,
                cache_dir=HF_CACHE_DIR
            )
        else:
            if self.local_files_only:
                cache_model_name = "models--" + self.model_name.replace("/", "--")
                available_models = self.list_models_in_cache()
                raise ValueError(
                    f"The model {self.model_name} (cache name: {cache_model_name}) has not been found in the cache directory.\n"
                    f"Cache directory: {HF_CACHE_DIR}\n"
                    f"Available models: {', '.join(available_models)}"
                )
            else:
                self.download_model()
        
    def check_model_in_cache(self):
        # Convert model name from "org/model" to "models--org--model" format used in cache
        cache_model_name = "models--" + self.model_name.replace("/", "--")
        cache_path = os.path.join(HF_CACHE_DIR, cache_model_name)
        return os.path.exists(cache_path)
        
    def list_models_in_cache(self):
        """List all model directories in the cache, filtering out non-model entries."""
        if not os.path.exists(HF_CACHE_DIR):
            return []
        # Filter to only show model directories (starting with "models--")
        all_items = os.listdir(HF_CACHE_DIR)
        model_dirs = [item for item in all_items if item.startswith("models--")]
        return model_dirs

    def download_model(self):
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            local_files_only=False,
            cache_dir=HF_CACHE_DIR
        )
        self.tokenizer = AutoTokenizer.from_pretrained(
            self.model_name,
            local_files_only=False,
            cache_dir=HF_CACHE_DIR
        )

    def generate(self, prompt: str):
        inputs = self.tokenizer(prompt, return_tensors="pt")
        outputs = self.model.generate(**inputs)
        print(outputs)
        char_outputs = self.tokenizer.decode(outputs) # TODO: Fix this
        return char_outputs

@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option("--model-name", type=str, default="google/gemma-3-4b-it") # meta-llama/Meta-Llama-3-8B-Instruct
@click.option("--prompt", type=str, default="Hello, how are you?")
@click.option("--local-files-only", is_flag=True, default=False, help="Only use files from local cache, don't download")
def main(model_name, prompt, local_files_only):
    model_handler = ModelHandler(model_name, local_files_only=local_files_only)
    outputs = model_handler.generate(prompt)
    print(outputs)

if __name__ == "__main__":
    main()
