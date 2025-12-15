# Probe Runner

Run open-source models using vLLM with easily-extractable probe values.

Uses a vLLM fork designed for easy extraction of probe values at run-time.

## Quick Start

To install the appropriate [vLLM](https://github.com/lysandermawby/vllm/) fork and download the relevant dependencies, run the `setup.sh` script.
Note that this uses [uv package management](https://docs.astral.sh/uv/). 

```bash
chmod +x setup.sh
./setup.sh
```

To enable downloading models from huggingface and pulling from your huggingface cache, copy the `.env.example` file to `.env` and insert the appropriate values.

```bash
cp .env.example .env
# Edit .env to contain a HF_TOKEN with read access and the path (without variable expansion) to your huggingface cache
```

## Run Inference

To run inference on models using this framework, run the `inference.py` file.

```bash
uv run python inference.py --model-name "google/gemma-3-4b-it" --prompt "How are you doing today?"
```
