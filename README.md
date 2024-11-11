# Structured State-Space Model (SSM) Training and Evaluation

This repository contains the code for training and evaluating a Structured State-Space Model (SSM) - S4 for classification tasks. The model uses custom numerical stability techniques and includes early stopping to prevent overfitting. This Readme explains the methodology, code structure, and instructions for running the model.

## Methodology

The goal of this project is to implement a state-space model with several techniques for managing numerical stability and improving training performance, inspired by the structured state-space (S4) model. Below is a detailed explanation of the methodology used in the code:

### 1. **Data Preprocessing and Embedding**
   - The training, validation, and test data are read from CSV files, each containing tokens and their associated part-of-speech (POS) tags.
   - **Embeddings:** Tokens are converted to embeddings using a pre-trained embedding dictionary (`word2embedding`). If a word is not found in the dictionary, it is represented as a zero vector.
   - **Sliding Window Chunking:** To create a fixed sequence length for training, overlapping sliding windows are used to generate sequences of length 4, ensuring all sequences have consistent lengths.
   - **Label Encoding:** POS tags are converted into simplified categories: Nouns, Verbs, Adjectives/Adverbs, and Others. Each tag is then one-hot encoded to prepare for classification.

### 2. **Model Architecture and Parameters**
   - The SSM model uses matrices \( A \), \( B \), \( C_{\text{mat}} \), \( W \), and bias vector \( b \) initialized using **Xavier Initialization** for stable training. Delta, a scaling factor used in discretization, is also initialized as a vector with small positive values.
   - **Hyperparameters:** The main hyperparameters include the state and embedding dimensions, learning rate, maximum gradient norm, and regularization strength.

### 3. **Discretization of Matrix \( B \)**
   - The matrix \( B \) is discretized with the formula:
     \[
     B_{\text{discrete}} = (\Delta A)^{-1} \cdot (\exp(\Delta A) - I) \cdot (\Delta B)
     \]
   - **Numerical Approximation:** To avoid direct matrix inversion, which can be computationally expensive and numerically unstable, the code approximates the inverse by solving a linear system instead. Additionally, \( \exp(\Delta A) \) is calculated, and regularization is applied to improve numerical stability.

### 4. **Training with Gradient Clipping and Early Stopping**
   - **Gradient Clipping:** Gradients are clipped in every iteration to prevent exploding gradients, ensuring the gradient norm stays below a defined maximum threshold.
   - **Early Stopping:** If the validation accuracy does not improve over a set number of epochs (patience), the training stops, and the model reverts to the best parameters observed.
   - **Cross-Entropy Loss:** The model uses cross-entropy loss with softmax output to classify each sequence based on the one-hot encoded labels.

### 5. **Evaluation Metrics**
   - **Accuracy, Precision, and Recall:** For each class, the model computes precision, recall, and overall accuracy on the test set to evaluate performance. A confusion matrix is built to calculate these metrics for each class individually.

### 6. **Numerical Stability Techniques**
   - **Regularization and Clipping:** Several regularization techniques are applied, including:
     - Adding a small epsilon to denominators and logarithmic calculations to avoid division by zero.
     - Regularizing the matrix \( D_{\text{mat}} \) (scaled \( A \)) by adding \( \epsilon \) times the identity matrix.
     - Constraining \( \Delta \) to lie within a reasonable range to prevent large or very small values that could lead to instability.

## Code Structure

- **main.m**: Main training script that initializes parameters, performs the training loop with early stopping, and saves the best model.
- **test.m**: Test script that loads the trained model, evaluates it on the test set, and computes accuracy, precision, and recall for each class.
- **prep_data.m**:
  - `prep_data`: Prepares data by embedding tokens, applying label simplification, and generating overlapping chunks.
  - `one_hot`: Converts labels to one-hot encoded vectors.
  - `convert_tags`: Simplifies POS tags into broader categories.
  - `embed`: Embeds tokens using the pre-trained embeddings and manages unknown words.
  - `chunk_data`: Splits sequences into overlapping chunks.
  - `validate_model`: Validates the model on the validation set.
  - `compute_training_accuracy`: Computes the training accuracy using the best model parameters.
  - `evaluate_model`: Evaluates the model on the test set and computes precision and recall per class.

## Getting Started

### Prerequisites

- **MATLAB**: Ensure MATLAB is installed on your system.
- **Data Files**: Place the CSV files for `train_data`, `valid_data`, `test_data`, and `embeddings` in the `data/` directory.

### Running the Code

1. **Training the Model**:
   - Run `main.m` to train the model. This will initialize parameters, perform training with early stopping, and save the best model to `trained_model.mat`.
   - Output includes per-epoch loss and accuracy, along with validation loss and accuracy.

   ```matlab
   % In MATLAB command window
   main
   ```

2. **Testing the Model**:
   - Run `test.m` to load the saved model parameters and evaluate them on the test set.
   - It will display the test accuracy and precision and recall for each class.

   ```matlab
   % In MATLAB command window
   test
   ```

3. **Understanding the Outputs**:
   - Training and validation accuracy and loss per epoch are displayed in `main.m`.
   - `test.m` displays the final test accuracy, loss, and per-class precision and recall.

### Example Output

Sample output for `test.m`:

```
Test Loss: 1.1745, Test Accuracy: 79.30%
Class 1 - Precision: 64.95%, Recall: 95.14%
Class 2 - Precision: 77.05%, Recall: 55.47%
Class 3 - Precision: 48.72%, Recall: 1.31%
Class 4 - Precision: 95.14%, Recall: 86.76%
```

## Customizing the Model

- **Adjust Hyperparameters** in `main.m` to explore different learning rates, embedding dimensions, and regularization strengths.
- **Modify Patience for Early Stopping** to control how many epochs the training will wait for improvement before stopping.
- **Set Sequence Length** (`SEQ_LEN` in `prep_data`) for different chunk sizes if using a different input format.

## Notes on Numerical Stability

- **Discretization Approximation:** The model uses a linear system approach to approximate matrix inversion, which improves computational efficiency and numerical stability.
- **Gradient Clipping:** Applied in each training step to prevent exploding gradients.
- **Delta Constraining:** Delta is constrained to stay within a reasonable range, preventing extreme values that can lead to instability.
- **Regularization of \( D_{\text{mat}} \):** A small regularization term (epsilon) is added to the matrix before inversion to ensure it remains well-conditioned.

## Troubleshooting

- **NaN or Inf Values in Parameters**: If NaN or Inf values appear, check the hyperparameters, especially the learning rate, and ensure they are not too high.
- **Gradient Explosion**: Reduce the `max_grad_norm` or learning rate if gradient explosion occurs frequently.
- **Convergence Issues**: If the model is not converging, consider increasing the patience value or changing the regularization strength (`lambda`).

---

This README provides a comprehensive guide to running and understanding the code. For further assistance, feel free to refer to MATLAB documentation or consult online resources related to state-space models and numerical stability in deep learning. \
Acknowledgements: The preprocessing code which creates the sliding context windows borrows ideas from https://github.com/piyush-jena/pos_tagging_transformer/blob/main/prep_data.m which is used for POS Tagging for Transformers. 
