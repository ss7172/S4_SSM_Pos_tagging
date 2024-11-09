% main.m
% Training Script for the SSM Model with Early Stopping and Training Accuracy Computation

% Clear workspace and command window
clear;
clc;

%% Load and Prepare Data

% Load datasets
train_data = readtable('data/train_data.csv');
valid_data = readtable('data/valid_data.csv');
test_data = readtable('data/test_data.csv');
embeddings = readtable('data/embeddings.csv');

% Clean and prepare embeddings
cleaned_embeddings = cellfun(@(str_embeddings) ...
    str2num(regexprep(strrep(strrep(str_embeddings, '[', ''), ']', ''), '\s+', ' ')), ...
    embeddings.embedding, 'UniformOutput', false);
word2embedding = containers.Map(embeddings.word, cleaned_embeddings);

% Prepare data using 'prep_data.m' script
[train_embs, test_embs, valid_embs, Y_train, Y_valid, Y_test] = ...
    prep_data(train_data, valid_data, test_data, word2embedding);

%% Set Hyperparameters
N = 64;               % State dimension (number of hidden units)
D = 64;               % Embedding dimension (size of word embeddings)
C = 4;                % Number of classes
eta = 0.0001;         % Learning rate
num_epochs = 100;     % Maximum number of training epochs
epsilon = 1e-6;       % Small value for numerical stability
max_grad_norm = 1.0;  % Maximum gradient norm for clipping
lambda = 1e-4;        % Regularization strength (weight decay)
patience = 10;        % Early stopping patience

%% Initialize Model Parameters with Xavier Initialization
rng('default');        % For reproducibility

% Xavier initialization for A and B
limit_A = sqrt(6 / (N + N));
A = rand(N, N) * 2 * limit_A - limit_A;

limit_B = sqrt(6 / (N + D));
B = rand(N, D) * 2 * limit_B - limit_B;

% Xavier initialization for C_mat and W
limit_C = sqrt(6 / (D + N));
C_mat = rand(D, N) * 2 * limit_C - limit_C;

limit_W = sqrt(6 / (C + D));
W = rand(C, D) * 2 * limit_W - limit_W;

% Initialize biases
b = zeros(C, 1);        % Bias vector initialized to zero

% Initialize Delta with positive values
Delta = abs(randn(N, 1)) + 0.1;  % Ensure Delta has values >= 0.1

%% Initialize Variables for Early Stopping
best_val_accuracy = 0;   % Best validation accuracy observed
best_A = A;
best_B = B;
best_C_mat = C_mat;
best_Delta = Delta;
best_W = W;
best_b = b;
patience_counter = 0;    % Counter for early stopping

%% Training Loop with Early Stopping
num_samples = length(train_embs);

for epoch = 1:num_epochs
    fprintf('Epoch %d/%d\n', epoch, num_epochs);
    
    % Optionally shuffle training data to improve generalization
    % Uncomment the following lines if you wish to shuffle the data each epoch
    % shuffle_idx = randperm(num_samples);
    % train_embs = train_embs(shuffle_idx);
    % Y_train = Y_train(shuffle_idx);
    
    epoch_loss = 0;
    correct_train = 0;
    total_train = 0;
    
    for i = 1:num_samples
        % Extract the input sequence and labels
        X_seq = train_embs{i};    % Shape: (4, 64)
        Y_seq = Y_train{i};       % Shape: (4, 4)
        
        % Ensure sequence length is 4
        if size(X_seq, 1) < 4
            continue;  % Skip samples with insufficient length
        end
        
        % Transpose X_seq to match the expected dimensions (D, T)
        X_seq = X_seq';           % Now X_seq is (64, 4)
        
        % Initialize hidden states
        h = zeros(N, 5);          % h(:, t) for t = 0 to 4
        y = zeros(D, 4);          % Outputs y_t for t = 1 to 4
        
        %% Discretization Step
        % Scale rows of A by Delta using element-wise multiplication
        D_mat = Delta .* A;  % Element-wise multiplication (N x N)
        I_N = eye(N);
        
        % Regularization for numerical stability
        D_mat_reg = D_mat + epsilon * I_N;
        
        % Check condition number to ensure D_mat_reg is invertible
        cond_number = cond(D_mat_reg);
        if cond_number > 1e12
            warning('D_mat is ill-conditioned with condition number: %e at sample %d. Adjusting D_mat.', cond_number, i);
            % Adjust D_mat by increasing regularization
            D_mat_reg = D_mat + (cond_number * epsilon) * I_N;
        end
        
        % Compute matrix exponential
        A_d = expm(D_mat);
        
        % Solve linear system instead of matrix inversion for B_d
        RHS = (A_d - I_N) * (Delta .* B);  % Element-wise multiplication
        B_d = D_mat_reg \ RHS;             % Use backslash operator for numerical stability
        
        %% Forward Pass Through Time Steps t = 1 to 4
        for t = 1:4
            x_t = X_seq(:, t);       % Input at time t (64, 1)
            h(:, t+1) = A_d * h(:, t) + B_d * x_t;
            y(:, t) = C_mat * h(:, t+1);
        end
        
        %% Output Layer Computation at Time t = 4
        logits = W * y(:, 4) + b;   % Shape: (4, 1)
        
        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits);
        hat_y = exp_logits / sum_exp;           % Shape: (4, 1)
        
        % Compute cross-entropy loss
        y_true = Y_seq(4, :)';                 % True labels at time t = 4 (4, 1)
        loss = -sum(y_true .* log(hat_y + epsilon));  % Add small epsilon for stability
        epoch_loss = epoch_loss + loss;
        
        %% Backward Pass
        % Gradient w.r.t logits
        dL_dlogits = hat_y - y_true;          % Shape: (4, 1)
        
        % Gradients w.r.t W and b
        dL_dW = dL_dlogits * y(:, 4)';        % Shape: (4, 64)
        dL_db = dL_dlogits;                   % Shape: (4, 1)
        
        % Gradient w.r.t y_4
        dL_dy4 = W' * dL_dlogits;             % Shape: (64, 1)
        
        % Gradient w.r.t C_mat and h_4
        dL_dC_mat = dL_dy4 * h(:, 5)';        % Shape: (64, 64)
        dL_dh = zeros(N, 5);                  % Initialize gradients w.r.t h_t
        dL_dh(:, 5) = C_mat' * dL_dy4;        % Gradient at t = 4
        
        % Backpropagation through time for h_t
        dL_dA_d = zeros(N, N);
        dL_dB_d = zeros(N, D);
        
        for t_step = 4:-1:1
            % Gradients w.r.t A_d and B_d
            dL_dA_d = dL_dA_d + dL_dh(:, t_step+1) * h(:, t_step)';
            dL_dB_d = dL_dB_d + dL_dh(:, t_step+1) * X_seq(:, t_step)';
            
            % Update gradient w.r.t h_t
            dL_dh(:, t_step) = A_d' * dL_dh(:, t_step+1);
        end
        
        %% Gradients w.r.t Original Parameters
        % Gradient w.r.t A
        dA_dA = Delta .* A_d;  % Element-wise multiplication
        dL_dA = dL_dA_d .* dA_dA;
        
        % Gradient w.r.t B
        dL_dB = dL_dB_d .* Delta;  % Element-wise multiplication
        
        % Gradient w.r.t Delta
        dL_dDelta = zeros(N, 1);
        for i_delta = 1:N
            % Accumulate gradients directly without using dA_dDelta_i and dB_dDelta_i
            dL_dDelta(i_delta) = sum(dL_dA_d(i_delta, :) .* A(:, i_delta)') + sum(dL_dB_d(i_delta, :) .* B(:, i_delta)');
        end
        
        %% Gradient Clipping
        % Compute total norm of gradients
        total_norm = 0;
        gradient_list = {dL_dA, dL_dB, dL_dC_mat, dL_dDelta, dL_dW, dL_db};
        for grad = gradient_list
            total_norm = total_norm + sum(grad{1}(:).^2);
        end
        total_norm = sqrt(total_norm);
        
        % Compute scaling factor
        scaling_factor = min(1, max_grad_norm / (total_norm + epsilon));
        
        % Scale gradients
        dL_dA = dL_dA * scaling_factor;
        dL_dB = dL_dB * scaling_factor;
        dL_dC_mat = dL_dC_mat * scaling_factor;
        dL_dDelta = dL_dDelta * scaling_factor;
        dL_dW = dL_dW * scaling_factor;
        dL_db = dL_db * scaling_factor;
        
        %% Apply Regularization (Weight Decay)
        dL_dA = dL_dA + lambda * A;
        dL_dB = dL_dB + lambda * B;
        dL_dC_mat = dL_dC_mat + lambda * C_mat;
        dL_dW = dL_dW + lambda * W;
        % Biases typically not regularized
        
        %% Parameter Updates
        A = A - eta * dL_dA;
        B = B - eta * dL_dB;
        C_mat = C_mat - eta * dL_dC_mat;
        Delta = Delta - eta * dL_dDelta;
        W = W - eta * dL_dW;
        b = b - eta * dL_db;
        
        % Clip Delta to prevent negative or excessively large values
        Delta = min(max(Delta, 0.1), 10.0);
        
        %% Check for NaN or Inf in Parameters
        if any(isnan(A(:))) || any(isinf(A(:))) || ...
           any(isnan(B(:))) || any(isinf(B(:))) || ...
           any(isnan(C_mat(:))) || any(isinf(C_mat(:))) || ...
           any(isnan(W(:))) || any(isinf(W(:))) || ...
           any(isnan(b(:))) || any(isinf(b(:))) || ...
           any(isnan(Delta(:))) || any(isinf(Delta(:)))
            error('NaN or Inf detected in parameters at sample %d, epoch %d', i, epoch);
        end
        
        % Compute Training Accuracy Incrementally
        [~, predicted_label] = max(hat_y);
        [~, true_label] = max(y_true);
        if predicted_label == true_label
            correct_train = correct_train + 1;
        end
        total_train = total_train + 1;
        
        % Display progress every 1000 samples to reduce verbosity
        if mod(i, 1000) == 0
            fprintf('Processed %d/%d samples\n', i, num_samples);
        end
    end
    
    %% Compute Average Loss and Training Accuracy for the Epoch
    avg_epoch_loss = epoch_loss / total_train;
    training_accuracy = correct_train / total_train;
    fprintf('Epoch %d completed. Average Loss: %.4f, Training Accuracy: %.2f%%\n', epoch, avg_epoch_loss, training_accuracy * 100);
    
    %% Validate the Model on the Validation Set
    [validation_loss, validation_accuracy] = validate_model(valid_embs, Y_valid, A, B, C_mat, Delta, W, b, epsilon);
    fprintf('Validation Loss: %.4f, Validation Accuracy: %.2f%%\n', validation_loss, validation_accuracy * 100);
    
    %% Early Stopping Logic
    if validation_accuracy > best_val_accuracy
        best_val_accuracy = validation_accuracy;
        % Save the best parameters
        best_A = A;
        best_B = B;
        best_C_mat = C_mat;
        best_Delta = Delta;
        best_W = W;
        best_b = b;
        patience_counter = 0;
        fprintf('Validation accuracy improved. Saving best model.\n');
    else
        patience_counter = patience_counter + 1;
        fprintf('Validation accuracy did not improve. Patience counter: %d/%d\n', patience_counter, patience);
        if patience_counter >= patience
            fprintf('Early stopping triggered. Best Validation Accuracy: %.2f%%\n', best_val_accuracy * 100);
            % Restore best model parameters
            A = best_A;
            B = best_B;
            C_mat = best_C_mat;
            Delta = best_Delta;
            W = best_W;
            b = best_b;
            break;  % Exit the training loop
        end
    end
end

%% Restore Best Parameters (if early stopping was triggered)
if patience_counter >= patience
    fprintf('Restored best model parameters based on validation accuracy.\n');
end

%% Compute Final Training Accuracy with Best Parameters
[training_loss_final, training_accuracy_final] = compute_training_accuracy(train_embs, Y_train, A, B, C_mat, Delta, W, b, epsilon);
fprintf('Final Training Loss: %.4f, Final Training Accuracy: %.2f%%\n', training_loss_final, training_accuracy_final * 100);

%% Save the Trained Model Parameters
save('trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');

%% End of Script

%% Helper Function Definitions

function [validation_loss, validation_accuracy] = validate_model(valid_embs, Y_valid, A, B, C_mat, Delta, W, b, epsilon)
    % Function to evaluate the model on the validation set
    % Inputs:
    %   valid_embs - cell array of validation embeddings
    %   Y_valid - cell array of validation labels
    %   A, B, C_mat, Delta, W, b - model parameters
    %   epsilon - small value for numerical stability
    %
    % Outputs:
    %   validation_loss - average cross-entropy loss on validation set
    %   validation_accuracy - accuracy on validation set

    num_samples = length(valid_embs);
    total_loss = 0;
    correct = 0;
    total = 0;

    for i = 1:num_samples
        % Extract the input sequence and labels
        X_seq = valid_embs{i};    % Shape: (4, 64)
        Y_seq = Y_valid{i};       % Shape: (4, 4)

        % Ensure sequence length is 4
        if size(X_seq, 1) < 4
            continue;  % Skip samples with insufficient length
        end

        % Transpose X_seq to match the expected dimensions (D, T)
        X_seq = X_seq';           % Now X_seq is (64, 4)

        % Initialize hidden states
        h = zeros(size(A, 1), 5);          % h(:, t) for t = 0 to 4
        y = zeros(size(C_mat, 1), 4);      % Outputs y_t for t = 1 to 4

        %% Discretization Step
        D_mat = Delta .* A;  % Element-wise multiplication (N x N)
        I_N = eye(size(A, 1));

        % Regularization for numerical stability
        D_mat_reg = D_mat + epsilon * I_N;

        % Check condition number to ensure D_mat_reg is invertible
        cond_number = cond(D_mat_reg);
        if cond_number > 1e12
            warning('D_mat is ill-conditioned with condition number: %e at validation sample %d. Adjusting D_mat.', cond_number, i);
            % Adjust D_mat by increasing regularization
            D_mat_reg = D_mat + (cond_number * epsilon) * I_N;
        end

        % Compute matrix exponential
        A_d = expm(D_mat);

        % Solve linear system instead of matrix inversion for B_d
        RHS = (A_d - I_N) * (Delta .* B);
        B_d = D_mat_reg \ RHS;  % Use backslash operator for numerical stability

        %% Forward Pass Through Time Steps t = 1 to 4
        for t = 1:4
            x_t = X_seq(:, t);       % Input at time t (64, 1)
            h(:, t+1) = A_d * h(:, t) + B_d * x_t;
            y(:, t) = C_mat * h(:, t+1);
        end

        %% Output Layer Computation at Time t = 4
        logits = W * y(:, 4) + b;   % Shape: (4, 1)

        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits);
        hat_y = exp_logits / sum_exp;           % Shape: (4, 1)

        % Compute cross-entropy loss
        y_true = Y_seq(4, :)';                 % True labels at time t = 4 (4, 1)
        loss = -sum(y_true .* log(hat_y + epsilon));  % Add small epsilon for stability
        total_loss = total_loss + loss;

        % Compute accuracy
        [~, predicted_label] = max(hat_y);
        [~, true_label] = max(y_true);
        if predicted_label == true_label
            correct = correct + 1;
        end
        total = total + 1;
    end

    % Compute average loss and accuracy
    validation_loss = total_loss / total;
    validation_accuracy = correct / total;
end

function [training_loss, training_accuracy] = compute_training_accuracy(train_embs, Y_train, A, B, C_mat, Delta, W, b, epsilon)
    % Function to compute training loss and accuracy using the best model parameters
    % Inputs:
    %   train_embs - cell array of training embeddings
    %   Y_train - cell array of training labels
    %   A, B, C_mat, Delta, W, b - best model parameters
    %   epsilon - small value for numerical stability
    %
    % Outputs:
    %   training_loss - average cross-entropy loss on training set
    %   training_accuracy - accuracy on training set

    num_samples = length(train_embs);
    total_loss = 0;
    correct = 0;
    total = 0;

    for i = 1:num_samples
        % Extract the input sequence and labels
        X_seq = train_embs{i};    % Shape: (4, 64)
        Y_seq = Y_train{i};       % Shape: (4, 4)

        % Ensure sequence length is 4
        if size(X_seq, 1) < 4
            continue;  % Skip samples with insufficient length
        end

        % Transpose X_seq to match the expected dimensions (D, T)
        X_seq = X_seq';           % Now X_seq is (64, 4)

        % Initialize hidden states
        h = zeros(size(A, 1), 5);          % h(:, t) for t = 0 to 4
        y = zeros(size(C_mat, 1), 4);      % Outputs y_t for t = 1 to 4

        %% Discretization Step
        D_mat = Delta .* A;  % Element-wise multiplication (N x N)
        I_N = eye(size(A, 1));

        % Regularization for numerical stability
        D_mat_reg = D_mat + epsilon * I_N;

        % Check condition number to ensure D_mat_reg is invertible
        cond_number = cond(D_mat_reg);
        if cond_number > 1e12
            warning('D_mat is ill-conditioned with condition number: %e at training sample %d. Adjusting D_mat.', cond_number, i);
            % Adjust D_mat by increasing regularization
            D_mat_reg = D_mat + (cond_number * epsilon) * I_N;
        end

        % Compute matrix exponential
        A_d = expm(D_mat);

        % Solve linear system instead of matrix inversion for B_d
        RHS = (A_d - I_N) * (Delta .* B);
        B_d = D_mat_reg \ RHS;  % Use backslash operator for numerical stability

        %% Forward Pass Through Time Steps t = 1 to 4
        for t = 1:4
            x_t = X_seq(:, t);       % Input at time t (64, 1)
            h(:, t+1) = A_d * h(:, t) + B_d * x_t;
            y(:, t) = C_mat * h(:, t+1);
        end

        %% Output Layer Computation at Time t = 4
        logits = W * y(:, 4) + b;   % Shape: (4, 1)

        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits);
        hat_y = exp_logits / sum_exp;           % Shape: (4, 1)

        % Compute cross-entropy loss
        y_true = Y_seq(4, :)';                 % True labels at time t = 4 (4, 1)
        loss = -sum(y_true .* log(hat_y + epsilon));  % Add small epsilon for stability
        total_loss = total_loss + loss;

        % Compute accuracy
        [~, predicted_label] = max(hat_y);
        [~, true_label] = max(y_true);
        if predicted_label == true_label
            correct = correct + 1;
        end
        total = total + 1;
    end

    % Compute average loss and accuracy
    training_loss = total_loss / total;
    training_accuracy = correct / total;
end
