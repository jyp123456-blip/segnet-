# SegNet Semantic Segmentation Model

## Features
- Encoder-decoder architecture with attention modules (MASAG, MOGA, CAS)
- Preserves max-pooling indices for accurate unpooling
- Suitable for image segmentation tasks

## Quick Use
```python
# Initialize model (21 classes)
model = SegNet(num_classes=21)

# Training
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters())

# Inference
with torch.no_grad():
    seg_map = torch.argmax(model(img_tensor), dim=1).squeeze()
```

## Dependencies
- Python 3.6+
- PyTorch 1.8+
- NumPy

## Architecture
- **Encoder**: 5 blocks with Conv-BN-ReLU, MASAG attention, max-pooling
- **Decoder**: 5 blocks with unpooling, Conv-BN-ReLU, MOGA attention
- **Output**: Final classification layer
