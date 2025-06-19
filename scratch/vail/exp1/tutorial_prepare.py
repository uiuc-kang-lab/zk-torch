#!/usr/bin/env python3
"""
Prepares the necessary files for the zk-torch tutorial.

This script performs the following actions:
1.  Creates a simple PyTorch model.
2.  Exports the model to ONNX format.
3.  Generates dummy input data.
4.  Saves the input data to a JSON file in the format expected by the Rust application.
5.  Prints the next steps for the user to follow.
"""
import torch
import torch.nn as nn
import json
import numpy as np
import os

# Define a very simple model (e.g., a single linear layer)
class SimpleModel(nn.Module):
    def __init__(self, input_features, output_features):
        super().__init__()
        self.linear = nn.Linear(input_features, output_features)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.linear(x)
        x = self.relu(x)
        return x

# --- Configuration ---
INPUT_FEATURES = 10
OUTPUT_FEATURES = 8
BATCH_SIZE = 1 # Keep batch size 1 for simplicity unless model handles batches specifically
SCRATCH_DIR = "scratch/vail/exp1/setup"
ONNX_MODEL_PATH = os.path.join(SCRATCH_DIR, "model.onnx")
INPUT_JSON_PATH = os.path.join(SCRATCH_DIR, "input.json")
# ---

# Ensure scratch directory exists
os.makedirs(SCRATCH_DIR, exist_ok=True)

# Instantiate the model
model = SimpleModel(INPUT_FEATURES, OUTPUT_FEATURES)

# Initialize weights to small random integer values to avoid possible all-zero edge case.
# The values are cast to float32, as expected by the model.
with torch.no_grad():
    model.linear.weight.data = torch.randint(-1, 2, model.linear.weight.shape).float()
    model.linear.bias.data = torch.randint(-1, 2, model.linear.bias.shape).float()

model.eval() # Set to evaluation mode

# Create dummy input data
# Input shape should match the model's expected input (batch_size, input_features)
# Using very small integer values to minimize output
dummy_input_np = np.random.randint(0, 2, (BATCH_SIZE, INPUT_FEATURES)).astype(np.float32)
dummy_input_torch = torch.from_numpy(dummy_input_np)

print(f"Dummy input shape: {dummy_input_torch.shape}")
print(f"Dummy input values: {dummy_input_np.flatten()}")

# Run the model to get expected output (for debugging)
with torch.no_grad():
    expected_output = model(dummy_input_torch)
    expected_output_np = expected_output.numpy()

print(f"Expected output shape: {expected_output_np.shape}")
print(f"Expected output values: {expected_output_np.flatten()}")

# Export the model to ONNX
print(f"Exporting model to {ONNX_MODEL_PATH}...")
torch.onnx.export(
    model,
    dummy_input_torch,
    ONNX_MODEL_PATH,
    input_names=['input'], # Optional: names for inputs
    output_names=['output'] # Optional: names for outputs
)
print("Model exported successfully.")

# Prepare input data for the Rust application's JSON format
# Expected format: {"input_data": [[flat_tensor_vals]]}
# Convert the numpy array to a flattened list of floats (f64 for Rust's JSON parser)
input_data_flat = dummy_input_np.flatten().tolist()

# Create the JSON structure
# The outer list corresponds to the number of inputs the ONNX model has (usually 1)
json_data = {
    "input_data": [input_data_flat]
}

# Save the JSON file
print(f"Saving input data to {INPUT_JSON_PATH}...")
with open(INPUT_JSON_PATH, 'w') as f:
    json.dump(json_data, f, indent=2)
print("Input data saved successfully.")

print(f"\n✅ Tutorial files generated in '{SCRATCH_DIR}/' directory:")
print(f"   - ONNX Model: {ONNX_MODEL_PATH}")
print(f"   - Input JSON: {INPUT_JSON_PATH}")
print("\nNext steps:")
# The config file path is relative to the project root, where the cargo command is run.
CONFIG_PATH = "scratch/vail/exp1/tutorial_config.yaml"
print(f"1. Verify that the paths in '{CONFIG_PATH}' are correct:")
print(f"   onnx.model_path: {ONNX_MODEL_PATH}")
print(f"   onnx.input_path: {INPUT_JSON_PATH}")
print(f"2. Update 'ptau.ptau_path' in '{CONFIG_PATH}' to point to a valid Power-of-Tau file.")
print("\n3. Run proof generation. This will create a proof file (e.g., scratch/vail/exp1/setup/proof.dat):")
print(f"   cargo run --release --bin prove -- {CONFIG_PATH}")
print("\n4. Verify the generated proof:")
print(f"   cargo run --release --bin verify -- {CONFIG_PATH}")
