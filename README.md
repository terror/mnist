## nn

**nn** is a neural network written from scratch with the goal of being able to
recognize handwritten digits, written in pure Rust with minimal dependencies.

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
[00:07:26] =====>---------------------------------- 7/50 Epochs Accuracy: 92.18%
```

### Prior Art

Wrote this while reading [**Make Your Own Neural Network**](https://www.amazon.ca/Make-Your-Own-Neural-Network/dp/1530826608) by Tariq Rashid.
