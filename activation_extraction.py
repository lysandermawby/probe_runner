#!/usr/bin/env python3
"""Activation extraction system for vLLM inference."""

import threading

import torch

# Lazy import to avoid loading vLLM modules before model initialization
# This prevents issues with vLLM's registry subprocess mechanism
_ActivationStore = None
_ActivationHookManager = None


def _lazy_import_activation_hooks():
    """Lazy import of activation hooks to avoid premature vLLM module loading."""
    global _ActivationStore, _ActivationHookManager
    if _ActivationStore is None or _ActivationHookManager is None:
        # Use importlib to handle the import robustly with the shim package
        # The shim re-exports attributes but submodule paths should still resolve
        import importlib
        module = importlib.import_module("vllm.v1.worker.activation_hooks")
        _ActivationStore = module.ActivationStore
        _ActivationHookManager = module.ActivationHookManager
    return _ActivationStore, _ActivationHookManager


class ActivationExtractor:
    """
    Main interface for activation extraction during inference.
    Stores activations for use in probe computation.
    """
    
    def __init__(self, extract_layers, enabled=True):
        """Initialize the activation extractor"""
        self.enabled = enabled
        self.extract_layers = extract_layers
        # Don't create ActivationStore here - delay until needed
        self.activation_store = None
        self.hook_manager = None
        self._model = None
        self._lock = threading.Lock()
    
    def register_model(self, model):
        """Register hooks on a model for activation extraction."""
        if not self.enabled:
            return
        
        # Lazy import and initialization of activation store
        ActivationStore, ActivationHookManager = _lazy_import_activation_hooks()
        
        with self._lock:
            # Initialize activation store if not already done
            if self.activation_store is None:
                self.activation_store = ActivationStore()
            
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
        # Lazy import and initialization if needed
        if self.activation_store is None:
            ActivationStore, _ = _lazy_import_activation_hooks()
            self.activation_store = ActivationStore()
        return self.activation_store
    
    def clear_activations(self, request_id=None):
        """Clear stored activations."""
        if self.activation_store is None:
            return
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
            if self.activation_store is not None:
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

