# Transformer Encoder Model with Custon Attention Mechanism:

the file ('encoder_torch.py') contaiuns the implementation for the Encoder Model. The core feature of this model is it's custom built attention mechanism which is implemented from scratch.

### Key Features:
- Custom Attention Mechanism implemented using Linear Layers
- Transformer Encoder Architecture utilizing the generalized Transformer encoder architecture which is suitable for a wide range of sequence-to-sequence and sequence labeling tasks
- Scalibility and Flexibility
- Performance Optimization using Layer Normalizations and Dropout to enhance training stability and Generalizing Learning whilist preventing overfitting.

### Implementation Detail:
- Layer Configuration: Each Layer contains multi-head attention
- Customizability: Model Parameters such as number of layers, number of attention heads, drop-out probability, Dimension of Key, Query & Value Matrices are easily configurable, allowing the user to tailor the model to specific requirements.


--------------------------------------------------------------

# How to Train:

1) Populate 'input' and 'target' arrays in 'train.py'
2) Adjust Model Parameters to suit your need
3) Adjust number of 'Epochs'
4) Run ' python3 train.py' in terminal

