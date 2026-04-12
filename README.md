# Image-ML: Multimodal Machine Learning Project

A comprehensive machine learning project combining image and text data for crisis informative content classification using ResNet-50 and BERT transformers.

## 📋 Table of Contents

- [Features](#features)
- [Project Structure](#project-structure)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Components](#project-components)
- [Usage](#usage)
- [Model Architecture](#model-architecture)
- [Results](#results)
- [Contributing](#contributing)
- [License](#license)

---

## ✨ Features

- **Multimodal Learning**: Combines image and text data for enhanced predictions
- **Pre-trained Models**: Uses ResNet-50 for images and BERT for text
- **Transfer Learning**: Leverages pre-trained weights for faster convergence
- **Data Processing**: Automatic image resizing, normalization, and text tokenization
- **GPU Support**: Full CUDA support for faster training
- **Poetry Dependency Management**: Professional package management with lock files
- **Jupyter Notebook**: Interactive training and evaluation pipeline

---

## 📁 Project Structure

```
Image-ML/
├── README.md                          # Project documentation
├── requirements.txt                   # Pip dependencies
├── pyproject.toml                     # Poetry configuration
├── poetry.lock                        # Locked dependency versions
├── DSL_imageGen.ipynb                 # Main training notebook
├── data/
│   ├── images/                        # Image dataset
│   └── crisismmd_datasplit_all/       # Text data splits
│       ├── task_informative_text_img_train.tsv
│       ├── task_informative_text_img_dev.tsv
│       └── task_informative_text_img_test.tsv
└── models/
    ├── image.pth                      # Pre-trained ResNet weights
    └── text.pth                       # Pre-trained BERT weights
```

---

## 📦 Prerequisites

- **Python**: 3.8 or higher
- **GPU**: CUDA 11.0+ (optional, but recommended for faster training)
- **Git**: For version control
- **Poetry**: For dependency management (recommended)

---

## 🚀 Installation

### Option 1: Using Poetry (Recommended)

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-ML.git
cd Image-ML

# Install Poetry (if not already installed)
pip install poetry

# Install dependencies
poetry install

# Activate virtual environment
poetry shell
```

### Option 2: Using pip

```bash
# Clone the repository
git clone https://github.com/yourusername/Image-ML.git
cd Image-ML

# Create virtual environment
python -m venv venv

# Activate virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

---

## ⚡ Quick Start

### 1. Prepare Your Data

Place your dataset in the `data/` directory:
```
data/
├── images/                    # Image files (.jpg, .png, etc.)
└── crisismmd_datasplit_all/   # TSV files with labels and text
    ├── task_informative_text_img_train.tsv
    ├── task_informative_text_img_dev.tsv
    └── task_informative_text_img_test.tsv
```

### 2. Prepare Your Models (Optional)

If using pre-trained weights, place them in `models/`:
```
models/
├── image.pth    # ResNet-50 weights
└── text.pth     # BERT weights
```

### 3. Run the Notebook

```bash
# Launch Jupyter
jupyter notebook

# Open DSL_imageGen.ipynb
# Follow the cells in order:
# 1. Import libraries
# 2. Set up configuration
# 3. Load and explore data
# 4. Train image model (ResNet-50)
# 5. Train text model (BERT)
# 6. Train multimodal model
# 7. Evaluate results
```

### 4. Train the Model

Execute cells in `DSL_imageGen.ipynb`:
- Configure hyperparameters (batch size, learning rate, epochs)
- Run training loops
- Monitor validation accuracy
- Save trained models

---

## 🏗️ Project Components

### Image Model: ResNet-50
- **Architecture**: ResNet-50 with custom fully connected layer
- **Input**: Images (224x224)
- **Output**: 2-class predictions (informative/non-informative)
- **Frozen**: Yes (transfer learning)

```python
class ResNetWithFC(nn.Module):
    def __init__(self, num_classes, hidden_dim=512):
        self.fc1 = nn.Linear(2048, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, num_classes)
```

### Text Model: BERT
- **Architecture**: BERT base uncased
- **Input**: Tweet text (max 64 tokens)
- **Output**: 2-class predictions
- **Fine-tuned**: Yes

```python
self.bert = BertModel.from_pretrained('bert-base-uncased')
```

### Multimodal Model: Fusion Network
- **Combines**: ResNet features (2) + BERT features (768) = 770 features
- **Architecture**: 3-layer fully connected network
- **Layers**: FC(770→256) → ReLU → FC(256→128) → ReLU → FC(128→2) → Softmax

```python
class ImageTextModel(nn.Module):
    def forward(self, input_ids, attention_masks, image_tensors):
        bert_output = self.bert(input_ids, attention_mask=attention_masks)[1]
        resnet_output = self.resnet(image_tensors)
        combined = torch.cat((bert_output, resnet_output), dim=1)
        return self.fc_layers(combined)
```

---

## 📊 Usage

### Training Image Model Only

```python
train_loader = DataLoader(train_data, batch_size=32, shuffle=True)
model = ResNetWithFC(num_classes=2)
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=0.0001)
criterion = nn.CrossEntropyLoss()

for epoch in range(num_epochs):
    for images, labels in train_loader:
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Training Text Model Only

```python
model_bert = BertForSequenceClassification.from_pretrained('bert-base-uncased', num_labels=2)
model_bert.to(device)
optimizer = torch.optim.Adam(model_bert.parameters(), lr=2e-5)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids = batch[0].to(device)
        attention_masks = batch[1].to(device)
        labels = batch[2].to(device)
        
        outputs = model_bert(input_ids, attention_mask=attention_masks, labels=labels)
        loss = outputs.loss
        loss.backward()
        optimizer.step()
```

### Training Multimodal Model

```python
model = ImageTextModel()
model.to(device)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-5)

for epoch in range(num_epochs):
    for batch in train_dataloader:
        input_ids = batch[0].squeeze(dim=1).to(device)
        attention_masks = batch[1].squeeze(dim=1).to(device)
        images = batch[2].to(device)
        labels = batch[3].to(device)
        
        outputs = model(input_ids, attention_masks, images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
```

### Evaluation

```python
model.eval()
correct = 0
total = 0

with torch.no_grad():
    for batch in val_dataloader:
        input_ids = batch[0].squeeze(dim=1).to(device)
        attention_masks = batch[1].squeeze(dim=1).to(device)
        images = batch[2].to(device)
        labels = batch[3].to(device)
        
        outputs = model(input_ids, attention_masks, images)
        _, predicted = torch.max(outputs.data, 1)
        total += labels.size(0)
        correct += (predicted == labels).sum().item()

accuracy = 100 * correct / total
print(f'Validation Accuracy: {accuracy:.2f}%')
```

---

## 🧠 Model Architecture

### Data Flow

```
Input Image (224x224)
    ↓
ResNet-50 (Frozen)
    ↓
FC Layer → [2-dim output]
    ↓
        ┌─────────────────────┐
        │  Concatenate (770)  │
        │  (2 + 768 features) │
        └─────────────────────┘
        ↓
Input Text (max 64 tokens)
    ↓
BERT Base Uncased
    ↓
CLS Token → [768-dim output]
    ↓
        ↓
    FC(770→256) + ReLU
        ↓
    FC(256→128) + ReLU
        ↓
    FC(128→2) + Softmax
        ↓
    Output: [Informative, Non-Informative]
```

---

## 📈 Training Configuration

| Parameter | Value |
|-----------|-------|
| Batch Size | 32 |
| Learning Rate | 1e-5 |
| Optimizer | Adam |
| Loss Function | CrossEntropyLoss |
| Epochs | 5 |
| Image Size | 224x224 |
| Max Text Length | 64 tokens |
| Device | CUDA (if available) |

---

## 📊 Results

After training, you'll see:
- **Training Loss**: Decreases over epochs
- **Validation Accuracy**: Typically 75-85% on crisis informative detection
- **Test Accuracy**: Final model performance

Example output:
```
Epoch 1/5 - Train Loss: 0.5234, Train Acc: 0.7150, Test Loss: 0.4821, Test Acc: 0.7890
Epoch 2/5 - Train Loss: 0.3892, Train Acc: 0.8220, Test Loss: 0.3654, Test Acc: 0.8450
Epoch 3/5 - Train Loss: 0.2954, Train Acc: 0.8760, Test Loss: 0.3102, Test Acc: 0.8790
Epoch 4/5 - Train Loss: 0.2341, Train Acc: 0.9010, Test Loss: 0.2987, Test Acc: 0.8820
Epoch 5/5 - Train Loss: 0.1876, Train Acc: 0.9240, Test Loss: 0.2845, Test Acc: 0.8950
```

---

## 🔄 Dependency Management

### Update Dependencies

```bash
# Using Poetry
poetry update

# Using pip
pip install --upgrade -r requirements.txt
```

### Add New Dependency

```bash
# Using Poetry
poetry add package-name

# Using pip
pip install package-name
pip freeze > requirements.txt
```

### Lock Dependencies

```bash
poetry lock
```

---

## 🛠️ Troubleshooting

### Issue: CUDA Out of Memory
**Solution**: Reduce batch size in notebook
```python
BATCH_SIZE = 16  # Reduce from 32
```

### Issue: Module Not Found Error
**Solution**: Reinstall dependencies
```bash
poetry install
# or
pip install -r requirements.txt
```

### Issue: Image Not Found
**Solution**: Verify image paths match TSV file structure
```python
# Check data directory
import os
print(os.listdir('data/images'))
```

### Issue: BERT Model Download Fails
**Solution**: Pre-download model
```python
from transformers import BertModel, BertTokenizer
model = BertModel.from_pretrained('bert-base-uncased')
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
```

---

## 📝 Notebook Structure

| Cell | Content |
|------|---------|
| 1 | Remove Colab code (local setup) |
| 2-3 | Import libraries & set paths |
| 4-6 | Load and explore data |
| 7-10 | Build ImageDataset class |
| 11-18 | Train ResNet image model |
| 19-24 | Train BERT text model |
| 25-30 | Build ImageTextModel (multimodal) |
| 31-40 | Train multimodal model |
| 41+ | Evaluation & results |

---

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'Add amazing feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Open Pull Request

---

## 📄 License

This project is licensed under the MIT License - see LICENSE file for details.

---

## 👤 Author

**Robin**
- Email: robin@example.com
- GitHub: [@yourusername](https://github.com/yourusername)

---

## 🔗 References

- [PyTorch Documentation](https://pytorch.org/docs/)
- [Transformers Library](https://huggingface.co/transformers/)
- [ResNet Paper](https://arxiv.org/abs/1512.03385)
- [BERT Paper](https://arxiv.org/abs/1810.04805)

---

## 📞 Support

For issues, questions, or suggestions:
1. Check existing GitHub issues
2. Create a new issue with detailed description
3. Include error messages and system info

---

**Last Updated**: April 12, 2026  
**Python Version**: 3.8+  
**Status**: Active Development ✅