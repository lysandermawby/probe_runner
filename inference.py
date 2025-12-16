#!/usr/bin/env python3
"""Load and run a model using vLLM with optional activation extraction"""

# package imports 
# import vllm
from models import ModelHandler
from activation_extraction import compute_probe, load_probe_from_file
import click


@click.command(context_settings=dict(help_option_names=['-h', '--help'], show_default=True))
@click.option("--model-name", type=str, default="google/gemma-3-4b-it")
@click.option("--prompt", type=str, default="Hello, how are you?")
@click.option("--extract-activations", is_flag=True, default=False,
              help="Enable activation extraction during inference")
@click.option("--extract-layers", type=str, default=None,
              help="Comma-separated list of layer indices to extract activations from (e.g., '0,5,10'). "
                   "If not specified and --extract-activations is set, extracts from all layers.")
@click.option("--probe-path", type=str, default=None,
              help="Path to probe weights file to compute probe outputs")
@click.option("--probe-layer", type=int, default=None,
              help="Layer index to use for probe computation (required if --probe-path is set)")
def main(model_name, prompt, extract_activations, extract_layers, probe_path, probe_layer):
    """Run inference with optional activation extraction and probe computation."""
    
    # Parse extract_layers if provided
    extract_layers_list = None
    if extract_layers:
        try:
            extract_layers_list = [int(x.strip()) for x in extract_layers.split(',')]
        except ValueError:
            click.echo("Error: --extract-layers must be comma-separated integers", err=True)
            return
    
    # Validate probe options
    if probe_path and probe_layer is None:
        click.echo("Error: --probe-layer is required when --probe-path is set", err=True)
        return
    
    # Initialize model with activation extraction if requested
    model = ModelHandler(
        model_name=model_name,
        extract_activations=extract_activations,
        extract_layers=extract_layers_list,
    )
    
    # Run generation
    generated_text = model.generate(prompt=prompt)
    print(f"Generated text: {generated_text}\n")
    
    # Handle activation extraction and probe computation
    if extract_activations:
        activation_store = model.get_activation_store()
        if activation_store:
            stats = activation_store.get_stats()
            print(f"Activation extraction stats: {stats}")
            
            # Show available layers and positions
            if stats["num_requests"] > 0:
                request_id = stats["requests"][0]
                layers = activation_store.get_layers_for_request(request_id)
                positions = activation_store.get_positions_for_request(request_id)
                print(f"Request {request_id}:")
                print(f"  Available layers: {sorted(layers)}")
                print(f"  Available positions: {sorted(positions)}")
                
                # Compute probe if requested
                if probe_path and probe_layer is not None:
                    try:
                        probe_weights, probe_bias = load_probe_from_file(probe_path)
                        print(f"\nLoaded probe from {probe_path}")
                        print(f"Probe weights shape: {probe_weights.shape}")
                        
                        # Get activation for the specified layer
                        # Use the first available position
                        if positions:
                            position = sorted(positions)[0]
                            activation = activation_store.get_activation(
                                request_id, probe_layer, position
                            )
                            
                            if activation is not None:
                                probe_output = compute_probe(activation, probe_weights, probe_bias)
                                print(f"\nProbe output for layer {probe_layer}, position {position}:")
                                print(f"  Value: {probe_output.item() if probe_output.numel() == 1 else probe_output}")
                            else:
                                print(f"\nWarning: No activation found for layer {probe_layer}, position {position}")
                        else:
                            print("\nWarning: No token positions available for probe computation")
                    except Exception as e:
                        print(f"\nError computing probe: {e}")
                else:
                    print("\nTip: Use --probe-path and --probe-layer to compute probe outputs")
        else:
            print("Warning: Activation extraction was requested but activation store is not available")


if __name__ == "__main__":
    main()
