#!/usr/bin/env python3
"""
Model handling powered by vLLM
Smart model downloading and inference methods, optimising available hardware
"""

# package imports
import os
from vllm import LLM, SamplingParams
import click # for CLI
from dotenv import load_dotenv
import torch # for device checking - native dependency of vLLM

class ModelHandler:
    """load, download, and run models using vllm"""
    def __init__(
        self,
        model_name,
        local_files_only=False,
        device='cuda',
        extract_activations=False,
        extract_layers=None,
    ):
        """initialise the ModelHandler including loading the model
        
        extracted_activations: If this is True, all activations will be extracted from the model. Otherwise, no activations will be extracted.
        extract_layers: List of layer indices to extract activations from
        extract_activations: If this is True, all activations will be  from the model. Otherwise, no activations will be extracted.
        """
        self.local_files_only = local_files_only
        self.model_name = model_name
        self.extract_activations = extract_activations
        self.extract_layers = extract_layers
        
        # activation extractor isn't created until after model initialisation
        self.activation_extractor = None
        
        self.set_environ()
        self.set_huggingface_cache()
        self.load_model()

    def set_environ(self):
        """Set environment variables needed for vLLM operation."""
        # Fix tokenizers parallelism warnings
        os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
        # Workaround for V1 engine serialization bug on CPU/macOS
        # This allows pickle-based serialization as a fallback
        os.environ.setdefault("VLLM_ALLOW_INSECURE_SERIALIZATION", "1")
        # os.environ['VLLM_LOGGING_LEVEL'] = 'ERROR'  # suppress large numbers of logs from the vLLM instance


    def set_huggingface_cache(self):
        """setting the HF_CACHE_DIR variable based on the .env file """
        load_dotenv()

        # default huggingface cache
        default_hf_home = "~/.cache/huggingface"
        # default_hf_cache = os.path.join(os.path.expanduser(default_hf_home), "hub")

        # Set HuggingFace cache directory to use local cache (only if not already set)
        if "HF_HOME" not in os.environ:
            os.environ["HF_HOME"] = os.path.expanduser(default_hf_home)

        # Get the cache directory - this is where the actual model files are stored
        # (HF_HOME/hub, not just HF_HOME)
        HF_CACHE_DIR = os.path.join(
            os.environ.get("HF_HOME", os.path.expanduser(default_hf_home)),
            "hub"
        )
        self.HF_CACHE_DIR = HF_CACHE_DIR


    def detect_device(self):
        """finding the available device"""
        if torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"


    def load_model(self):
        """load model using vLLM"""
        if self.local_files_only and not self.check_model_in_cache():
            available_models = self.list_models_in_cache()
            raise ValueError(
                f"Model {self.model_name} not found in cache (local_files_only=True).\n"
                f"Cache directory: {self.HF_CACHE_DIR}\n"
                f"Available models: {', '.join(available_models)}"
            )

        # will download the model if needed
        print(f"Loading model {self.model_name}")
        if not self.check_model_in_cache():
            print(f"Model not in cache, downloading to {self.HF_CACHE_DIR}")
        
        self.model = LLM(
            model = self.model_name,
            download_dir = self.HF_CACHE_DIR
        )
        
        # Now that model is loaded, we can safely import and create activation extractor
        # This prevents vLLM registry subprocess issues
        if self.extract_activations:
            # Lazy import - only import after model is loaded
            from activation_extraction import ActivationExtractor
            self.activation_extractor = ActivationExtractor(
                extract_layers=self.extract_layers,
                enabled=True,
            )
        
        # Register activation extraction hooks if enabled
        if self.activation_extractor is not None:
            # Access the underlying model through the engine
            engine = self.model.llm_engine
            
            # Try the new get_model() method (available in recent versions)
            # This is the recommended approach going forward
            if hasattr(engine, 'get_model'):
                model = engine.get_model()
                if model is not None:
                    self.activation_extractor.register_model(model)
                    print("Registered activation extraction hooks on model (via get_model)")
            
            # Legacy path for V0 engine and older versions
            elif hasattr(engine, 'model_executor'):
                # For V0 engine with model_executor
                model_executor = engine.model_executor
                
                # Try to get driver_worker
                if hasattr(model_executor, 'driver_worker'):
                    worker = model_executor.driver_worker
                    if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                        model = worker.model_runner.model
                        self.activation_extractor.register_model(model)
                        print("Registered activation extraction hooks on model (via driver_worker)")
                # Try workers array as fallback
                elif hasattr(model_executor, 'workers'):
                    workers = model_executor.workers
                    if workers and len(workers) > 0:
                        worker = workers[0]
                        if hasattr(worker, 'model_runner') and hasattr(worker.model_runner, 'model'):
                            model = worker.model_runner.model
                            self.activation_extractor.register_model(model)
                            print("Registered activation extraction hooks on model (via workers[0])")
            
            # V1 engine path (note: V1 uses a different multiprocess architecture)
            # Direct model access is more limited in V1 due to the process separation
            else:
                print("Warning: Could not access model. No expected APIs found")


    def check_model_in_cache(self):
        """convert from 'org/model' HF format to 'models--org-model' format used in cache, and searching"""
        cache_model_name = self.hf_to_cache_format(self.model_name)
        cache_path = os.path.join(self.HF_CACHE_DIR, cache_model_name)
        return os.path.exists(cache_path)
    

    def list_models_in_cache(self):
        """list all model directories in the HF cache"""
        if not os.path.exists(self.HF_CACHE_DIR):
            return []
        
        all_items = os.listdir(self.HF_CACHE_DIR)
        model_dirs = [item for item in all_items if item.startswith("models--")]
        return model_dirs
    

    def hf_to_cache_format(self, model_name):
        """convert from 'org/model' HF format to 'models--org-model' format used in cache"""
        return "models--" + model_name.replace("/", "--")
    

    def generate(self, prompt, max_tokens=None, max_length=None, max_new_tokens=100,
        temperature=1.0, top_p=1.0, top_k=0, repetition_penalty=1.0, stop=None, **kwargs):
        """
        Running inference using transformers-like API.
        Activations are stored in self.activation_extractor.activation_store
        """
        # Handle max_length/max_new_tokens for transformers compatibility
        max_tokens = max_tokens or max_length or max_new_tokens

        # Build SamplingParams directly, matching notebook pattern exactly
        # Only include parameters that are explicitly set (not None for optional ones)
        params_dict = {
            "temperature": temperature,
            "top_p": top_p,
            "top_k": top_k,
            "repetition_penalty": repetition_penalty,
            "max_tokens": max_tokens,
        }
        
        if stop is not None:
            params_dict["stop"] = stop
        
        params_dict.update(kwargs)
        sampling_params = SamplingParams(**params_dict)

        # Set request context for activation extraction
        if self.activation_extractor is not None:
            # Generate a simple request ID for this generation
            request_id = f"req_{id(prompt)}"
            self.activation_extractor.set_request_context([request_id])

        outputs = self.model.generate([prompt], sampling_params)

        generated_text = outputs[0].outputs[0].text
        
        # Clear request context after generation
        if self.activation_extractor is not None:
            self.activation_extractor.clear_request_context()
    
        return generated_text
    
    def get_activation_store(self):
        """Get the activation store for probe computation."""
        if self.activation_extractor is not None:
            return self.activation_extractor.get_activation_store()
        return None



@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option("--model-name", type=str, default="google/gemma-3-4b-it") # alternative: meta-llama/Meta-Llama-3-8B-Instruct
@click.option("--prompt", type=str, default="Hello, how are you?")
@click.option("--local-files-only", is_flag=True, default=False, help="Only use files from local cache, don't download")
def main(model_name, prompt, local_files_only):
    model_handler = ModelHandler(model_name, local_files_only=local_files_only)
    outputs = model_handler.generate(prompt)
    print(outputs)


if __name__ == "__main__":
    main()
