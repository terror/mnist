## mnist

[![CI](https://github.com/terror/mnist/actions/workflows/ci.yml/badge.svg)](https://github.com/terror/mnist/actions/workflows/ci.yml)

**mnist** is a neural network written from scratch with the goal of being able to
recognize handwritten digits, written in pure Rust with minimal dependencies
used in the training process.

<div align="center">
  <img width="562" alt="Screenshot 2024-08-06 at 1 38 16 PM" src="https://github.com/user-attachments/assets/15c150e5-6e82-467b-8d19-b3176bc21e4e">
</div>

### Usage

Train the model and make predictions using saved weights.

```
Usage: mnist <COMMAND>

Commands:
  app      Run an interactive GUI application
  predict  Predict the class of a new instance
  train    Train the model
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

### Training

Below are the results from training on the standard MNIST dataset. This used a
learning rate of `0.1`, a batch size of `128`, and trained for `100` epochs. These
values are configurable via the `train` subcommand.

```
Dataset loaded successfully:
  Training images: 60000
  Training labels: 60000
  Test images: 10000
  Test labels: 10000
[01:24:29] ======================================== 100/100 Epochs Training complete
Final accuracy: 97.16%
Saved weights to weights.json
```

### Inference

You can use the `predict` subcommand with pre-defined weights to make
predictions on new data.

```bash
cargo run -- predict --weights weights.json --image samples/2.png
```

### Live drawing

You can spawn a GUI app for live drawing/inference using the CLI as well:

```
cargo run -- app --weights weights.json
```

The app is built using [`egui`](https://www.egui.rs/) and re-uses a neural net
with stored weights for making live predictions.

### Prior Art

Wrote this while reading [**Make Your Own Neural Network**](https://www.amazon.ca/Make-Your-Own-Neural-Network/dp/1530826608) by Tariq Rashid.
