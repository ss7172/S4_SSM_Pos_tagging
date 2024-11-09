% main_ad_updated.m
% Training Script for the SSM Model with Automatic Differentiation and Early Stopping

% Clear workspace and command window
clear;
clc;

%% Load and Prepare Data
% Assume you have the 'prep_data.m' script and data files in your working directory.

% Load data
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
N = 8;               % State dimension (number of hidden units)
D = 64;              % Embedding dimension (size of word embeddings)
C = 4;               % Number of classes
eta = 0.0001;        % Learning rate
num_epochs = 100;    % Maximum number of training epochs
epsilon = 1e-6;      % Small value for numerical stability
max_grad_norm = 1.0; % Maximum gradient norm for clipping
lambda = 1e-4;       % Regularization strength (weight decay)
patience = 10;       % Early stopping patience

%% Initialize Model Parameters with Xavier Initialization
rng('default');      % For reproducibility

% Xavier initialization for A and B
limit_A = sqrt(6 / (N + N));
A_init = rand(N, N) * 2 * limit_A - limit_A;

limit_B = sqrt(6 / (N + D));
B_init = rand(N, D) * 2 * limit_B - limit_B;

% Xavier initialization for C_mat and W
limit_C = sqrt(6 / (D + N));
C_init = rand(D, N) * 2 * limit_C - limit_C;

limit_W = sqrt(6 / (C + D));
W_init = rand(C, D) * 2 * limit_W - limit_W;

% Initialize biases
b_init = zeros(C, 1);        % Bias vector initialized to zero

% Initialize Delta with positive values
Delta_init = abs(randn(N, 1)) + 0.1;  % Ensure Delta has values >= 0.1

%% Convert Parameters to dlarray
A = dlarray(single(A_init), 'CB');         % 'C' - Channel (state units), 'B' - Batch (time steps)
B = dlarray(single(B_init), 'CB');         % 'C' - Channel (state units), 'B' - Batch (time steps)
C_mat = dlarray(single(C_init), 'CB');     % 'C' - Channel (output features), 'B' - Batch
W = dlarray(single(W_init), 'CB');         % 'C' - Channel (classes), 'B' - Batch
b = dlarray(single(b_init), 'C');          % 'C' - Channel (classes)
Delta = dlarray(single(Delta_init), 'C');  % 'C' - Channel (state units)

%% Initialize Velocities for Momentum (Optional)
% If you plan to use momentum, initialize velocities
velocity.A = zeros(size(A), 'like', A);
velocity.B = zeros(size(B), 'like', B);
velocity.C_mat = zeros(size(C_mat), 'like', C_mat);
velocity.W = zeros(size(W), 'like', W);
velocity.b = zeros(size(b), 'like', b);
velocity.Delta = zeros(size(Delta), 'like', Delta);

%% Training Loop with Automatic Differentiation and Early Stopping
num_samples = length(train_embs);
best_val_accuracy = 0;
best_params = struct(); % To store the best parameters
patience_counter = 0;

for epoch = 1:num_epochs
    fprintf('Epoch %d/%d\n', epoch, num_epochs);
    
    % We do not shuffle training data to preserve sequence order
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
        
        % Convert X_seq and Y_seq to dlarray
        dlX_seq = dlarray(single(X_seq), 'CB');          % 'C' - Channel (features), 'B' - Batch (time steps)
        dlY_seq = dlarray(single(Y_seq(4, :))', 'C');    % Only use labels at t = 4 (C x 1)
        
        %% Compute Loss and Gradients Using Automatic Differentiation
        [loss, gradients] = dlfeval(@modelLoss, A, B, C_mat, Delta, W, b, dlX_seq, dlY_seq, epsilon);
        epoch_loss = epoch_loss + extractdata(loss);
        
        %% Compute Training Accuracy
        % Forward pass to get predictions
        [~, pred_label] = max(extractdata(loss > 0)); % Not directly possible, need to compute predictions
        % Alternatively, compute predictions from logits
        % Modify modelLoss to return logits if needed
        % Here, we compute predictions separately
        
        % Recompute logits for prediction
        [logits, ~] = forwardPass(A, B, C_mat, Delta, W, b, dlX_seq, epsilon);
        [~, predicted_label] = max(extractdata(logits));
        [~, true_label] = max(extractdata(dlY_seq));
        
        if predicted_label == true_label
            correct_train = correct_train + 1;
        end
        total_train = total_train + 1;
        
        %% Gradient Clipping
        % Compute total norm of gradients
        grad_fields = fieldnames(gradients);
        total_norm = 0;
        for k = 1:numel(grad_fields)
            grad_field = grad_fields{k};
            grad_value = gradients.(grad_field);
            total_norm = total_norm + sum(extractdata(grad_value).^2, 'all');
        end
        total_norm = sqrt(total_norm);
        
        % Compute scaling factor
        scaling_factor = min(1, max_grad_norm / (total_norm + epsilon));
        
        % Scale gradients
        for k = 1:numel(grad_fields)
            grad_field = grad_fields{k};
            gradients.(grad_field) = gradients.(grad_field) * scaling_factor;
        end
        
        %% Apply Regularization (Weight Decay)
        gradients.A = gradients.A + lambda * A;
        gradients.B = gradients.B + lambda * B;
        gradients.C_mat = gradients.C_mat + lambda * C_mat;
        gradients.W = gradients.W + lambda * W;
        % Biases typically not regularized
        
        %% Update Parameters Using SGD with Momentum
        [A, velocity.A] = sgdMomentumUpdate(A, gradients.A, velocity.A, eta, velocity.A, 0.9);
        [B, velocity.B] = sgdMomentumUpdate(B, gradients.B, velocity.B, eta, velocity.B, 0.9);
        [C_mat, velocity.C_mat] = sgdMomentumUpdate(C_mat, gradients.C_mat, velocity.C_mat, eta, velocity.C_mat, 0.9);
        [Delta, velocity.Delta] = sgdMomentumUpdate(Delta, gradients.Delta, velocity.Delta, eta, velocity.Delta, 0.9);
        [W, velocity.W] = sgdMomentumUpdate(W, gradients.W, velocity.W, eta, velocity.W, 0.9);
        [b, velocity.b] = sgdMomentumUpdate(b, gradients.b, velocity.b, eta, velocity.b, 0.0); % No momentum for bias
        
        % Clip Delta to prevent negative or excessively large values
        Delta = max(Delta, 0.1);
        Delta = min(Delta, 10.0);
        
        %% Check for NaN or Inf in Parameters
        if any(isnan(extractdata(A))) || any(isinf(extractdata(A))) || ...
           any(isnan(extractdata(B))) || any(isinf(extractdata(B))) || ...
           any(isnan(extractdata(C_mat))) || any(isinf(extractdata(C_mat))) || ...
           any(isnan(extractdata(W))) || any(isinf(extractdata(W))) || ...
           any(isnan(extractdata(b))) || any(isinf(extractdata(b))) || ...
           any(isnan(extractdata(Delta))) || any(isinf(extractdata(Delta)))
            error('NaN or Inf detected in parameters at sample %d, epoch %d', i, epoch);
        end
    end
    
    %% Validation and Early Stopping
    % After each epoch, evaluate on validation set
    [validation_loss, validation_accuracy] = validateModel(A, B, C_mat, Delta, W, b, valid_embs, Y_valid, epsilon);
    fprintf('Validation Loss: %.4f, Validation Accuracy: %.2f%%\n', validation_loss, validation_accuracy * 100);
    
    % Check if validation accuracy improved
    if validation_accuracy > best_val_accuracy
        best_val_accuracy = validation_accuracy;
        % Save the best parameters
        best_params.A = A;
        best_params.B = B;
        best_params.C_mat = C_mat;
        best_params.Delta = Delta;
        best_params.W = W;
        best_params.b = b;
        patience_counter = 0;
        fprintf('Validation accuracy improved. Saving best model.\n');
    else
        patience_counter = patience_counter + 1;
        fprintf('Validation accuracy did not improve. Patience counter: %d/%d\n', patience_counter, patience);
        if patience_counter >= patience
            fprintf('Early stopping triggered. Best Validation Accuracy: %.2f%%\n', best_val_accuracy * 100);
            break;
        end
    end
end

%% Restore Best Parameters
A = best_params.A;
B = best_params.B;
C_mat = best_params.C_mat;
Delta = best_params.Delta;
W = best_params.W;
b = best_params.b;

%% Compute Training Accuracy with Best Parameters
[training_loss, training_accuracy] = computeTrainingAccuracy(A, B, C_mat, Delta, W, b, train_embs, Y_train, epsilon);
fprintf('Final Training Loss: %.4f, Final Training Accuracy: %.2f%%\n', training_loss, training_accuracy * 100);

%% Save the Trained Model Parameters
% Convert parameters back to double for saving
A = double(extractdata(A));
B = double(extractdata(B));
C_mat = double(extractdata(C_mat));
Delta = double(extractdata(Delta));
W = double(extractdata(W));
b = double(extractdata(b));

save('trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');

%% End of Script

%% Function Definitions

function [loss, gradients] = modelLoss(A, B, C_mat, Delta, W, b, dlX_seq, dlY_seq, epsilon)
    % Forward pass with automatic differentiation
    % dlX_seq: dlarray of size [D x T]
    % dlY_seq: dlarray of size [C x 1]
    D = size(dlX_seq, 1);
    N = size(A, 1);
    C = size(W, 1);
    T = size(dlX_seq, 2);  % Should be 4
    
    % Initialize hidden states
    h = dlarray(zeros(N, T+1), 'CB');  % 'C' - Channel (state units), 'B' - Batch (time steps)
    y = dlarray(zeros(D, T), 'CB');    % 'C' - Channel (features), 'B' - Batch (time steps)
    
    %% Discretization Step
    % Compute D_mat = Delta .* A
    D_mat = Delta .* A;  % Element-wise multiplication (N x N)
    I_N = eye(N, 'like', A);  % Identity matrix (N x N)
    
    % Compute matrix exponential
    A_d = expm(D_mat);  % Discretized state transition matrix (N x N)
    
    % Compute B_d = D_mat_reg \ ((A_d - I) * (Delta .* B))
    RHS = (A_d - I_N) * (Delta .* B);  % (N x D)
    
    % Regularization for numerical stability
    D_mat_reg = D_mat + epsilon * I_N;
    
    % Compute B_d using backslash operator for numerical stability
    B_d = D_mat_reg \ RHS;  % (N x D)
    
    %% Forward Pass Through Time Steps t = 1 to T
    for t = 1:T
        x_t = dlX_seq(:, t);       % Input at time t (D x 1)
        h(:, t+1) = A_d * h(:, t) + B_d * x_t;
        y(:, t) = C_mat * h(:, t+1);
    end
    
    %% Output Layer Computation at Time t = T
    logits = W * y(:, T) + b;   % Shape: (C x 1)
    
    % Compute softmax probabilities
    logits_stable = logits - max(logits);   % For numerical stability
    exp_logits = exp(logits_stable);
    sum_exp = sum(exp_logits) + epsilon;
    hat_y = exp_logits / sum_exp;           % Shape: (C x 1)
    
    % Compute cross-entropy loss
    loss = -sum(dlY_seq .* log(hat_y + epsilon));
    
    %% Compute Gradients
    gradients = dlgradient(loss, {A, B, C_mat, Delta, W, b}, 'RetainData', true);
end

function [logits, h] = forwardPass(A, B, C_mat, Delta, W, b, dlX_seq, epsilon)
    % Forward pass to compute logits for prediction
    % Returns logits and hidden states
    D = size(dlX_seq, 1);
    N = size(A, 1);
    C = size(W, 1);
    T = size(dlX_seq, 2);  % Should be 4
    
    % Initialize hidden states
    h = dlarray(zeros(N, T+1), 'CB');  % 'C' - Channel (state units), 'B' - Batch (time steps)
    y = dlarray(zeros(D, T), 'CB');    % 'C' - Channel (features), 'B' - Batch (time steps)
    
    %% Discretization Step
    % Compute D_mat = Delta .* A
    D_mat = Delta .* A;  % Element-wise multiplication (N x N)
    I_N = eye(N, 'like', A);  % Identity matrix (N x N)
    
    % Compute matrix exponential
    A_d = expm(D_mat);  % Discretized state transition matrix (N x N)
    
    % Compute B_d = D_mat_reg \ ((A_d - I) * (Delta .* B))
    RHS = (A_d - I_N) * (Delta .* B);  % (N x D)
    
    % Regularization for numerical stability
    D_mat_reg = D_mat + epsilon * I_N;
    
    % Compute B_d using backslash operator for numerical stability
    B_d = D_mat_reg \ RHS;  % (N x D)
    
    %% Forward Pass Through Time Steps t = 1 to T
    for t = 1:T
        x_t = dlX_seq(:, t);       % Input at time t (D x 1)
        h(:, t+1) = A_d * h(:, t) + B_d * x_t;
        y(:, t) = C_mat * h(:, t+1);
    end
    
    %% Output Layer Computation at Time t = T
    logits = W * y(:, T) + b;   % Shape: (C x 1)
end

function [validation_loss, validation_accuracy] = validateModel(A, B, C_mat, Delta, W, b, valid_embs, Y_valid, epsilon)
    % Function to compute validation loss and accuracy
    num_samples = length(valid_embs);
    total_loss = 0;
    correct = 0;
    total = 0;
    
    for i = 1:num_samples
        X_seq = valid_embs{i};
        Y_seq = Y_valid{i};
        
        if size(X_seq, 1) < 4
            continue;
        end
        
        X_seq = X_seq';
        dlX_seq = dlarray(single(X_seq), 'CB');
        dlY_seq = dlarray(single(Y_seq(4, :))', 'C');  % Only t=4
        
        % Forward pass to get logits
        [logits, ~] = forwardPass(A, B, C_mat, Delta, W, b, dlX_seq, epsilon);
        
        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits) + epsilon;
        hat_y = exp_logits / sum_exp;           % Shape: (C x 1)
        
        % Compute cross-entropy loss
        loss = -sum(dlY_seq .* log(hat_y + epsilon));
        total_loss = total_loss + extractdata(loss);
        
        % Compute accuracy
        [~, predicted_label] = max(extractdata(hat_y));
        [~, true_label] = max(extractdata(dlY_seq));
        if predicted_label == true_label
            correct = correct + 1;
        end
        total = total + 1;
    end
    
    validation_loss = total_loss / total;
    validation_accuracy = correct / total;
end

function [training_loss, training_accuracy] = computeTrainingAccuracy(A, B, C_mat, Delta, W, b, train_embs, Y_train, epsilon)
    % Function to compute training loss and accuracy
    num_samples = length(train_embs);
    total_loss = 0;
    correct = 0;
    total = 0;
    
    for i = 1:num_samples
        X_seq = train_embs{i};
        Y_seq = Y_train{i};
        
        if size(X_seq, 1) < 4
            continue;
        end
        
        X_seq = X_seq';
        dlX_seq = dlarray(single(X_seq), 'CB');
        dlY_seq = dlarray(single(Y_seq(4, :))', 'C');  % Only t=4
        
        % Forward pass to get logits
        [logits, ~] = forwardPass(A, B, C_mat, Delta, W, b, dlX_seq, epsilon);
        
        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits) + epsilon;
        hat_y = exp_logits / sum_exp;           % Shape: (C x 1)
        
        % Compute cross-entropy loss
        loss = -sum(dlY_seq .* log(hat_y + epsilon));
        total_loss = total_loss + extractdata(loss);
        
        % Compute accuracy
        [~, predicted_label] = max(extractdata(hat_y));
        [~, true_label] = max(extractdata(dlY_seq));
        if predicted_label == true_label
            correct = correct + 1;
        end
        total = total + 1;
    end
    
    training_loss = total_loss / total;
    training_accuracy = correct / total;
end

function [param, velocity] = sgdMomentumUpdate(param, grad, velocity, learning_rate, current_velocity, momentum)
    % Update parameters using SGD with momentum
    velocity = momentum * velocity - learning_rate * grad;
    param = param + velocity;
end
