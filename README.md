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

### Speed Up Reinstalls

First-time setup will compile the CUDA kernels from the vLLM fork, which can take between 15 minutes to 2 hours depending on your hardware.
To speedup reinstalls, you can backup the compiled extensions by creating a tarball of the new vllm instance.

```bash
tar -czf vllm_compiled_$(nvidia-smi --query-gpu=name --format=csv,noheader | head -1 | tr ' ' '_')_cuda$(nvcc --version | grep release | sed 's/.*release //' | sed 's/,.*//').tar.gz vllm/
```

Downloading this tarball and using it in future when running inference with this vllm fork will avoid recompiling the kernels.

## Run Inference

To run inference on models using this framework, run the `inference.py` file.

```bash
uv run python inference.py --model-name "google/gemma-3-4b-it" --prompt "How are you doing today?"
```
