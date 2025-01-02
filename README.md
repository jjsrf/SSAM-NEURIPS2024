# SAM.v2: One-Step Sharpness-Aware Minimization

## Introduction to SAM

Sharpness-Aware Minimization (SAM) is an optimization technique that simultaneously minimizes loss value and loss sharpness. The traditional SAM requires two forward-backward passes to compute a single update, which introduces significant computational overhead.

## Our Method: Single-Step SAM

Our modified version (SAM.v2) maintains the benefits of the original SAM while reducing computational overhead through:

1. **Gradient Memory**: Storing and reusing previous gradients from the last iteration
2. **One-Step Update**: Combining two optimization steps into one efficient update
3. **State Preservation**: Maintaining parameter states between steps without extra forward passes

### Key Features

- **One-Step Optimization**: SAM.v2 reduces computational overhead by storing and reusing previous gradients
- **Adaptive Support**: Optional adaptive SAM implementation for better performance on some tasks
- **Flexible Integration**: Works with various optimizers and model architectures
- **Memory Efficient**: Maintains state between steps without requiring additional forward passes

## Code Implementation

### SAM Optimizer Definition

The SAM optimizer is defined in `sam.py`. Here's how to initialize it in `main_prune_train.py`:

```python
elif args.optmzr == 'sgd-sam':
    base_optimizer = torch.optim.SGD
    optimizer = SAM(model.parameters(), 
                   base_optimizer, 
                   rho=args.sam_rho,           # Neighborhood size
                   adaptive=args.adaptive,      # Enable adaptive SAM
                   v2=args.sam_v2,             # Enable one-step SAM.v2
                   lr=args.lr,                 
                   momentum=args.momentum, 
                   weight_decay=args.weight_decay)
```

### Basic Usage

```python
from sam import SAM

# Initialize SAM optimizer
optimizer = SAM(
    model.parameters(),
    base_optimizer,
    rho=0.05,              # Neighborhood size
    adaptive=False,        # Whether to use adaptive SAM
    v2=True,              # Enable one-step SAM.v2
    lr=0.1,               # Learning rate
    momentum=0.9,         # Momentum parameter
    weight_decay=5e-4     # Weight decay
)

# Training loop
optimizer.zero_grad()
optimizer.first_step(zero_grad=True)
loss = criterion(model(inputs), targets)
loss.backward()
optimizer.second_step(zero_grad=True)
```

### Command Line Arguments

When using the training script (`main_prune_train.py`), the following SAM-related arguments are available:

```bash
--optmzr sgd-sam        # Use SAM optimizer
--sam-rho 0.05         # Set SAM's rho parameter
--sam-v2               # Enable one-step SAM.v2
--adaptive            # Enable adaptive SAM
```

### Example Training Command

```bash
python main_prune_train.py \
    --arch resnet \
    --depth 20 \
    --dataset cifar10 \
    --optmzr sgd-sam \
    --sam-v2 \
    --sam-rho 0.05 \
    --lr 0.1 \
    --epochs 160
```

## Requirements

- PyTorch >= 1.0
- torchvision
- CUDA (optional, but recommended for faster training)

## Citation

If you find this implementation useful, please cite our work:

```
@inproceedings{jisingle,
  title={A Single-Step, Sharpness-Aware Minimization is All You Need to Achieve Efficient and Accurate Sparse Training},
  author={Ji, Jie and Li, Gen and Fu, Jingjing and Afghah, Fatemeh and Guo, Linke and Yuan, Xiaoyong and Ma, Xiaolong},
  booktitle={The Thirty-eighth Annual Conference on Neural Information Processing Systems}
}
```
