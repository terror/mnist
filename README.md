## nn

[![CI](https://github.com/terror/nn/actions/workflows/ci.yml/badge.svg)](https://github.com/terror/nn/actions/workflows/ci.yml)

**nn** is a neural network written from scratch with the goal of being able to
recognize handwritten digits, written in pure Rust with minimal dependencies
used in the training process.

<img width="1667" alt="Screenshot 2024-08-05 at 2 42 44 PM" src="https://github.com/user-attachments/assets/341a5bc9-383b-4186-84e5-fc579945555b">

### Usage

Train the model and make predictions using saved weights.

```
Usage: nn <COMMAND>

Commands:
  train
  predict
  help     Print this message or the help of the given subcommand(s)

Options:
  -h, --help  Print help
```

### Training

Below are the results from training on the standard MNIST dataset. This used a
learning rate of `0.1`, a batch size of `128`, and trained for `50` epochs. These
values are configurable via the `train` subcommand.

```
Dataset loaded successfully:
  Training images: 60000
  Training labels: 60000
  Test images: 10000
  Test labels: 10000
[00:53:14] ======================================== 50/50 Epochs Training complete
Final accuracy: 96.45%
Saved weights to weights.json
```

### Inference

You can use the `predict` subcommand with pre-defined weights to make
predictions on new data.

```bash
cargo run -- predict --weights weights.json --image samples/2.png
```

### Prior Art

Wrote this while reading [**Make Your Own Neural Network**](https://www.amazon.ca/Make-Your-Own-Neural-Network/dp/1530826608) by Tariq Rashid.
