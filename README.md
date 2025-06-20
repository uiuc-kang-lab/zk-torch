# Zk-Torch

## Overview
Zk-Torch is a library for generating ZK circuits from ONNX models with the feature of Mira-style folding. We support all edge models in MLPerf Inference.

## Prerequisites
### Install Rust
```
curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs | sh
```
### Install nightly Rust
```
rustup override set nightly
```

## Run the example
```
cargo run --release --bin zk_torch --features fold -- config.yaml
```

## How to run custom experiments
- Replace the model and input in `config.yaml`
    ```
    model_path: <path_to_your_model>
    input_path: <path_to_your_input>
    ```
- Use customized scale factor
    ```
    scale_factor_log: <log2 of the scale factor>
    ```
