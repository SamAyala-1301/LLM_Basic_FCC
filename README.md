# Bigram Language Model Chatbot (GPT-based)

### Overview

This project implements a basic chatbot built on a custom bigram language model using PyTorch. The architecture draws from transformer principles like self-attention, multi-head attention, and layer normalization. It is trained on the Wizard of Oz dataset, generating responses based on the input prompt.

### Key Features

	•	Transformer-based Architecture: Uses a decoder-only transformer.
	•	Bigram Model: Processes character-level bigrams (two-character sequences).
	•	Self-Attention Mechanism: Learns dependencies between characters in sequences.
	•	Text Generation: Can generate new tokens based on a given context.

### Model Architecture

1. Embedding Layers

	•	Token Embedding Table: Maps input characters to a learned embedding of size n_embd=384.
	•	Position Embedding Table: Adds positional information to token embeddings, allowing the model to capture the order of input sequences.

2. Multi-Head Self-Attention

	•	Heads: Each attention head (4 heads in total) projects the input into key, query, and value vectors.
	•	Scaled Dot-Product Attention: The queries and keys are used to compute attention weights, which are then multiplied by the values to capture long-range dependencies between tokens.
	•	Causal Masking: Ensures the model only attends to prior tokens when generating new tokens, avoiding “leakage” from future information.

3. FeedForward Layers

	•	Fully Connected Layers: Each attention block is followed by a two-layer feedforward network with a hidden dimension of 4 * n_embd and ReLU activation. This layer helps transform and mix the attention outputs.

4. Layer Normalization

	•	Applied twice: once after multi-head attention and once after the feedforward network. Layer normalization helps stabilize training by normalizing across the feature dimension.

5. Final Linear Layer

	•	After passing through all attention layers, the final logits are generated via a linear projection. These logits represent the likelihood of each token in the vocabulary.

6. Loss Function

	•	Cross-Entropy Loss: The model calculates loss using cross-entropy between predicted and target tokens for each time step in the sequence. This is used to backpropagate and optimize the weights.

### Hyperparameters

	•	Block Size: 32 (the maximum context length for the model).
	•	Batch Size: 128 (samples processed in parallel).
	•	Embedding Dimension (n_embd): 384.
	•	Number of Attention Heads: 4.
	•	Number of Layers: 4 transformer decoder layers.
	•	Dropout: 0.2 (regularization to prevent overfitting).
	•	Learning Rate: 2e-5 (controlled via AdamW optimizer).
	•	Max Iterations: 200 (number of training iterations).

### Data Processing

	•	Dataset: The training corpus is the Wizard of Oz text.
	•	Character-Level Encoding: Text is tokenized at the character level. Each character is mapped to an integer index, and the sequence is split into training (80%) and validation (20%) data.
	•	Bigram Modeling: The model predicts the next character given the previous two (bigram structure) and generates text in this manner.

### Functions Breakdown

1. get_batch():

Generates random batches of sequences (block_size tokens long) for training and validation. Each input batch is accompanied by the next character as a target.

2. estimate_loss():

Evaluates the model’s performance on training and validation data by calculating the average loss over multiple batches.

3. GPTLanguageModel:

Main class defining the model’s architecture. It handles token and positional embeddings, attention layers, and feedforward layers. The model can also generate new sequences given an initial context.

4. generate():

Generates text by iteratively predicting the next token and adding it to the input sequence, using a sampling method from the token distribution at each step.

### Future Work

	•	Model Optimization: Tuning hyperparameters (learning rate, dropout, embedding size) to improve generation quality.
	•	Better Dataset: Training on a larger, more coherent text corpus.
	•	Fine-Tuning: Introducing advanced training techniques like weight decay or scheduling to stabilize learning.

Troubleshooting

	•	If the model generates nonsensical or repetitive text, consider increasing the dataset size, adjusting hyperparameters, or training for more iterations.
	•	Be aware of potential overfitting, especially with smaller datasets.
