# ChatBot

A comprehensive chatbot implementation featuring two main components: a sequence-to-sequence conversational AI and a reading comprehension question-answering system.

## Overview

This repository contains two distinct but complementary AI systems:

1. **Seq2Seq + Attention Chatbot** - A conversational AI trained on movie dialogue data
2. **Reading Comprehension System** - A question-answering model for text comprehension tasks

## Project Structure

```
ChatBot/
├── seq2seq+attention/          # Conversational chatbot implementation
│   ├── ChatBot_main.py        # Main training and inference script
│   ├── chatbot_model.py       # Seq2seq model with attention mechanism
│   └── data_preprocessing.py  # Data processing for Cornell Movie Dialogs
└── Reading-Comprehension/     # Question-answering system
    ├── train.py              # Training script for QA model
    ├── qa_model.py           # QA model implementation
    ├── qa_data.py            # Data handling for QA tasks
    ├── evaluate.py           # Evaluation metrics (F1, exact match)
    ├── seq2seq_modify.py     # Modified seq2seq components
    ├── plot_data.py          # Data visualization utilities
    ├── figure_1.png          # Performance visualization
    └── preprocessing/        # Data preprocessing utilities
        ├── squad_preprocess.py
        └── dwr.py
```

## Features

### Seq2Seq + Attention Chatbot
- **Architecture**: Encoder-decoder with attention mechanism
- **Dataset**: Cornell Movie Dialogs Corpus
- **Framework**: TensorFlow
- **Key Features**:
  - Multi-layer LSTM networks
  - Attention mechanism for better context understanding
  - Bucketed training for efficient processing
  - Configurable vocabulary and model parameters

### Reading Comprehension System
- **Task**: Question answering on text passages
- **Dataset**: SQuAD (Stanford Question Answering Dataset)
- **Evaluation**: F1 score and exact match accuracy
- **Key Features**:
  - Bidirectional LSTM encoder
  - Attention-based answer prediction
  - Comprehensive evaluation metrics

## Getting Started

### Prerequisites

```bash
pip install tensorflow numpy
```

### Running the Seq2Seq Chatbot

1. **Prepare the data**: Ensure you have the Cornell Movie Dialogs Corpus in the specified directory structure
2. **Train the model**:
   ```python
   python seq2seq+attention/ChatBot_main.py
   ```

### Running the Reading Comprehension System

1. **Preprocess the data**:
   ```python
   python Reading-Comprehension/preprocessing/squad_preprocess.py
   ```

2. **Train the model**:
   ```python
   python Reading-Comprehension/train.py
   ```

3. **Evaluate performance**:
   ```python
   python Reading-Comprehension/evaluate.py
   ```

## Configuration

### Chatbot Configuration
Key parameters can be modified in [`data_preprocessing.py`](seq2seq+attention/data_preprocessing.py):

- `BATCH_SIZE`: Training batch size (default: 64)
- `HIDDEN_SIZE`: LSTM hidden units (default: 256)
- `NUM_LAYERS`: Number of LSTM layers (default: 3)
- `BUCKETS`: Sequence length buckets for efficient training
- `THRESHOLD`: Minimum word frequency threshold

### QA System Configuration
Training parameters in [`train.py`](Reading-Comprehension/train.py):

- `learning_rate`: Initial learning rate (default: 0.5)
- `batch_size`: Training batch size (default: 10)
- `state_size`: Model hidden state size (default: 200)
- `dropout`: Dropout rate (default: 0.15)

## Model Architecture

### Chatbot Model
- **Encoder**: Multi-layer bidirectional LSTM
- **Decoder**: Multi-layer LSTM with attention mechanism
- **Attention**: Bahdanau-style attention for context focusing
- **Output**: Softmax over vocabulary for next word prediction

### QA Model
- **Passage Encoder**: Bidirectional LSTM for context encoding
- **Question Encoder**: Bidirectional LSTM for question encoding
- **Answer Prediction**: Attention-based start/end position prediction
- **Loss Function**: Cross-entropy for position classification

## Evaluation

The reading comprehension system uses standard SQuAD evaluation metrics:
- **Exact Match (EM)**: Percentage of predictions that match ground truth exactly
- **F1 Score**: Token-level F1 score between prediction and ground truth

Results visualization is available through [`plot_data.py`](Reading-Comprehension/plot_data.py).

## Contributing

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Add tests if applicable
5. Submit a pull request

## License

This project is open source. Please check individual file headers for specific licensing information.

## Acknowledgments

- Cornell Movie Dialogs Corpus for conversational data
- Stanford Question Answering Dataset (SQuAD) for reading comprehension data
- TensorFlow team for the deep learning framework