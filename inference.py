#!/usr/bin/env python3
"""Load and run a model using vLLM"""

# package imports 
# import vllm
from models import ModelHandler
import click

@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option("--model-name", type=str, default="google/gemma-3-4b-it")
@click.option("--prompt", type=str, default="Hello, how are you?")
def main(model_name, prompt):
    model = ModelHandler(model_name=model_name)
    generated_text = model.generate(prompt=prompt)
    print(generated_text)


if __name__ == "__main__":
    main()
