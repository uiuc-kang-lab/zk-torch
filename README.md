# ZKTorch

  

## Overview

[Zero-knowledge (ZK) proofs of ML model inference](https://medium.com/@danieldkang/bridging-the-gap-how-zk-snarks-bring-transparency-to-private-ml-models-with-zkml-e0e59708c2fc) help provide transparency to users without requiring model owners to share model weights. Past work on these provers can be placed into two categories. The first method compiles the ML model into a low-level circuit, and the second method uses custom cryptographic protocols designed only for a specific class of models. Unfortunately, the first method is highly inefficient, and the second method does not generalize well.

ZKTorch is an end-to-end proving system for compiling ML model inference computation into ZK circuits from ONNX models by compiling layers into a set of specialized cryptographic operations, which we call basic blocks. It is built on top of a parallel extension to the Mira accumulation scheme, enabling succinct proofs with minimal accumulation overhead. We support all edge models in the [MLPerf Edge Inference Suite v4.1](https://github.com/mlcommons/inference_policies/blob/master/inference_rules.adoc#benchmarks-1), covering convolutional neural networks (CNNs), recurrent neural networks (RNNs), and large language models (LLMs). Overall, ZKTorch supports 61 layers with a total of 20 basic blocks. With the Mira accumulator extension, we condense proofs of the same basic block type.

![zk_torch_readme](https://github.com/user-attachments/assets/6715728d-1818-4ee2-9732-35fafc53976c)

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

The example runs ZKTorch with Mira-style folding enabled on the ONNX file and configurations specified in `config.yaml`. The ONNX file contains a small model with two fully-connected layers and two ReLU layers.

```
cargo run --release --bin zk_torch --features fold -- config.yaml
```

## How to run custom experiments

  

### Update the ptau file

Most models will require a larger powers of tau (ptau) file than the provided `challenge` file which has `pow_len_log=7`. The size of the ptau file needed (`pow_len_log` in `config.yaml`) depends on the magnitude of the quantized values to support in the inference computation (`cq_range_log` in `config.yaml`) as well as the sizes of the inputs to certain layers. In most cases, the former will be the deciding factor, with the constraint `cq_range_log` < `pow_len_log`.

To produce a larger file, please refer to the following instructions to generate one with the `snarkjs` tool:
https://github.com/iden3/snarkjs?tab=readme-ov-file#1-start-a-new-powers-of-tau-ceremony. For step 1, you can replace `14` with the desired value for the `pow_len_log` and then directly follow the remaining instructions through step 4 which produces the file with the `snarkjs powersoftau export challenge` command.

Then, update `config.yaml` based on the produced ptau file.
   
  ```
ptau_path: <path_to_ptau_file, e.g., challenge>
pow_len_log: <log of largest supported power>
loaded_pow_len_log: <log of largest power you want to load>
cq_range_log: <can be up to pow_len_log - 1>
cq_range_lower_log: <can be up to cq_range_log - 1>
  ```
For example, here is a valid configuration for `challenge_0003` produced by the example instructions:

  ```
ptau_path: challenge_0003
pow_len_log: 14
loaded_pow_len_log: 14
cq_range_log: 6
cq_range_lower_log: 5
  ```

### Replace the model and input in `config.yaml`
`model_path` should contain the path to the ONNX file to compile. `input_path` can be left blank or contain a JSON file similar to the example below, replacing the value with a tensor value.

`{"input_data":  [[0.09,  0.13,  0.24,  0.05]]}`

If it is left blank or the provided path does not exist, Zk-Torch will generate a random input tensor based on the input shape specified in the ONNX file, or otherwise throw an error.

Update the `config.yaml`:
```
model_path: <path_to_your_onnx_file>
input_path: <path_to_your_input>
```

### Use customized scale factor
Update `config.yaml` based on the desired quantization scale factor.
```
scale_factor_log: <log2 of the scale factor>
```
### Run experiment
If you change the ptau file after `layers_setup/`, `models`, and `setups` have been produced from a previous run, please delete them before proving.

Then run
```
cargo run --release --bin zk_torch --features fold -- <your config file>
```
To just run proving (e.g., for testing purposes), you can additionally add the `mock_prove` feature (`--features mock_prove,fold`).

The outputs consist of input, model (including weights and lookup tables), output, and setup encodings, as well as the proof before accumulation, accumulation-specific proofs, and the final proof after accumulation for the prover and verifier. The output paths are specified in `config.yaml`.
