#!/usr/bin/env python3
"""Activation extraction system for vLLM inference."""

import threading

import torch

from vllm.v1.worker.activation_hooks import ActivationStore, ActivationHookManager


class ActivationExtractor:
    """
    Main interface for activation extraction during inference.
    Stores activations for use in probe computation.
    """
    
    def __init__(self, extract_layers, enabled=True):
        """Initialize the activation extractor"""
        self.enabled = enabled
        self.extract_layers = extract_layers
        self.activation_store = ActivationStore()
        self.hook_manager = None
        self._model = None
        self._lock = threading.Lock()
    
    def register_model(self, model):
        """Register hooks on a model for activation extraction."""
        if not self.enabled:
            return
        
        with self._lock:
            if self.hook_manager is not None:
                self.hook_manager.remove_hooks()
            
            self._model = model
            self.hook_manager = ActivationHookManager(
                activation_store=self.activation_store,
                extract_layers=self.extract_layers,
            )
            self.hook_manager.register_hooks(model)
    
    def set_request_context(self, request_ids, token_positions=None):
        """Set the current request context for activation extraction."""
        if self.hook_manager is not None:
            self.hook_manager.set_request_context(request_ids, token_positions)
    
    def clear_request_context(self):
        """Clear the current request context."""
        if self.hook_manager is not None:
            self.hook_manager.clear_request_context()
    
    def get_activation_store(self):
        """Get the activation store for probe computation."""
        return self.activation_store
    
    def clear_activations(self, request_id=None):
        """Clear stored activations."""
        if request_id is not None:
            self.activation_store.clear_request(request_id)
        else:
            self.activation_store.clear_all()
    
    def cleanup(self):
        """Remove hooks and cleanup resources."""
        with self._lock:
            if self.hook_manager is not None:
                self.hook_manager.remove_hooks()
                self.hook_manager = None
            self.activation_store.clear_all()
            self._model = None
    
    def __enter__(self):
        """Context manager entry."""
        return self
    
    def __exit__(self, exc_type, exc_val, exc_tb):
        """Context manager exit."""
        self.cleanup()


def compute_probe(activation, probe_weights, probe_bias=None):
    """Compute probe output (dot product) for an activation."""
    if probe_weights.dim() == 1:
        # Single output probe
        output = torch.dot(activation, probe_weights)
        if probe_bias is not None:
            output = output + probe_bias
        return output
    else:
        # Multi-class probe
        output = torch.matmul(probe_weights, activation)
        if probe_bias is not None:
            output = output + probe_bias
        return output


def load_probe_from_file(probe_path):
    """Load a probe from a file."""
    checkpoint = torch.load(probe_path, map_location="cpu")
    
    if isinstance(checkpoint, dict):
        weights = checkpoint.get("weight", checkpoint.get("weights"))
        bias = checkpoint.get("bias", checkpoint.get("bias", None))
    else:
        # Assume it's just the weights
        weights = checkpoint
        bias = None
    
    if weights is None:
        raise ValueError(f"Could not find weights in probe file: {probe_path}")
    
    return weights, bias

