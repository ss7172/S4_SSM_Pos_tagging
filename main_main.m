% main.m
% Training Script for the SSM Model with Integrated Recommendations

% Clear workspace and command window
clear;
clc;

%% Load and Prepare Data

train_data = readtable('data/train_data.csv');
valid_data = readtable('data/valid_data.csv');
test_data = readtable('data/test_data.csv');
embeddings = readtable('data/embeddings.csv');

cleaned_embeddings = cellfun(@(str_embeddings) str2num(regexprep(strrep(strrep(str_embeddings, '[', ''), ']', ''), '\s+', ' ')), embeddings.embedding, 'UniformOutput', false);
word2embedding = containers.Map(embeddings.word, cleaned_embeddings);

[train_embs, test_embs, valid_embs, Y_train, Y_valid, Y_test] = prep_data(train_data, valid_data, test_data, word2embedding);

%% Set Hyperparameters
N = 64;               % State dimension (number of hidden units)
D = 64;                % Embedding dimension (size of word embeddings)
C = 4;                 % Number of classes
eta = 0.0001;          % Reduced learning rate for stability
num_epochs = 10;       % Number of training epochs
epsilon = 1e-6;        % Small value for numerical stability
max_grad_norm = 1.0;   % Maximum gradient norm for clipping
lambda = 1e-4;         % Regularization strength (weight decay)

%% Initialize Model Parameters with Xavier Initialization
rng('default');        % For reproducibility

A = rand(N, N);

B = rand(N, D);

C_mat = rand(D, N);

W = rand(C, D);

b = zeros(C, 1);       

Delta = abs(randn(N, 1)) + 0.1; 

num_samples = length(train_embs);

for epoch = 1:num_epochs
    fprintf('Epoch %d/%d\n', epoch, num_epochs);
    
    % We do not shuffle training data to preserve sequence order
    
    epoch_loss = 0;
    
    for i = 1:num_samples
        X_seq = train_embs{i};    % Shape: (4, 64)
        Y_seq = Y_train{i};       % Shape: (4, 4)
        
        if size(X_seq, 1) < 4
            continue;  % Skip samples with insufficient length
        end
        
        X_seq = X_seq';           % Now X_seq is (64, 4)
        
        h = zeros(N, 5);          % h(:, t) for t = 0 to 4
        y = zeros(D, 4);          % Outputs y_t for t = 1 to 4
        
        D_mat = diag(Delta) * A;
        I_N = eye(N);
        
        D_mat_reg = D_mat + epsilon * I_N;
        
        
        A_d = expm(D_mat);
        
        RHS = (A_d - I_N) * (diag(Delta) * B);
        B_d = D_mat_reg \ RHS;  
        
        for t = 1:4
            x_t = X_seq(:, t);       % Input at time t (64, 1)
            h(:, t+1) = A_d * h(:, t) + B_d * x_t;
            y(:, t) = C_mat * h(:, t+1);
        end
        
        logits = W * y(:, 4) + b;   % Shape: (4, 1)
        
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits);
        hat_y = exp_logits / sum_exp;           % Shape: (4, 1)
        
        y_true = Y_seq(4, :)';                 % True labels at time t = 4 (4, 1)
        loss = -sum(y_true .* log(hat_y + epsilon));  % Add small epsilon for stability
        epoch_loss = epoch_loss + loss;
        
        dL_dlogits = hat_y - y_true;          % Shape: (4, 1)
        
        dL_dW = dL_dlogits * y(:, 4)';        % Shape: (4, 64)
        dL_db = dL_dlogits;                   % Shape: (4, 1)
        
        dL_dy4 = W' * dL_dlogits;             % Shape: (64, 1)
        
        dL_dC = dL_dy4 * h(:, 5)';            % Shape: (64, N)
        dL_dh = zeros(N, 5);                  % Initialize gradients w.r.t h_t
        dL_dh(:, 5) = C_mat' * dL_dy4;        % Gradient at t = 4
        
        dL_dA_d = zeros(N, N);
        dL_dB_d = zeros(N, D);
        
        for t = 4:-1:1
            dL_dA_d = dL_dA_d + dL_dh(:, t+1) * h(:, t)';
            dL_dB_d = dL_dB_d + dL_dh(:, t+1) * X_seq(:, t)';
            
            dL_dh(:, t) = A_d' * dL_dh(:, t+1);
        end
        
        dA_dA = diag(Delta) * A_d;  % Derivative of A_d w.r.t A
        dL_dA = dL_dA_d .* dA_dA;
        
        dB_dB = diag(Delta);        % Since B_d depends linearly on B after scaling
        dL_dB = dL_dB_d .* (dB_dB * ones(N, D));
        
        dL_dDelta = zeros(N, 1);
        for i_delta = 1:N
            dA_dDelta_i = A(:, :) * A_d(i_delta, :)' .* Delta(i_delta);
            dB_dDelta_i = B(:, :) * Delta(i_delta);
            dL_dDelta(i_delta) = sum(sum(dL_dA_d .* dA_dDelta_i)) + sum(sum(dL_dB_d .* dB_dDelta_i));
        end
        
        total_norm = 0;
        gradient_list = {dL_dA, dL_dB, dL_dC, dL_dDelta, dL_dW, dL_db};
        for grad = gradient_list
            total_norm = total_norm + sum(grad{1}(:).^2);
        end
        total_norm = sqrt(total_norm);
        
        scaling_factor = min(1, max_grad_norm / (total_norm + epsilon));
        
        dL_dA = dL_dA * scaling_factor;
        dL_dB = dL_dB * scaling_factor;
        dL_dC = dL_dC * scaling_factor;
        dL_dDelta = dL_dDelta * scaling_factor;
        dL_dW = dL_dW * scaling_factor;
        dL_db = dL_db * scaling_factor;
        
        dL_dA = dL_dA + lambda * A;
        dL_dB = dL_dB + lambda * B;
        dL_dC = dL_dC + lambda * C_mat;
        dL_dW = dL_dW + lambda * W;
        A = A - eta * dL_dA;
        B = B - eta * dL_dB;
        C_mat = C_mat - eta * dL_dC;
        Delta = Delta - eta * dL_dDelta;
        W = W - eta * dL_dW;
        b = b - eta * dL_db;
        
        Delta = min(max(Delta, 0.1), 10.0);
        
        % if any(isnan(A(:))) || any(isinf(A(:))) || ...
        %    any(isnan(B(:))) || any(isinf(B(:))) || ...
        %    any(isnan(C_mat(:))) || any(isinf(C_mat(:))) || ...
        %    any(isnan(W(:))) || any(isinf(W(:))) || ...
        %    any(isnan(b(:))) || any(isinf(b(:))) || ...
        %    any(isnan(Delta(:))) || any(isinf(Delta(:)))
        %     error('NaN or Inf detected in parameters at sample %d, epoch %d', i, epoch);
        % end
        
        % Display progress every 1000 samples to reduce verbosity
        if mod(i, 1000) == 0
            fprintf('Processed %d/%d samples\n', i, num_samples);
        end
    end
    
    avg_epoch_loss = epoch_loss / num_samples;
    fprintf('Epoch %d completed. Average Loss: %.4f\n', epoch, avg_epoch_loss);
    
    
end

%% Save the Trained Model Parameters
save('trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');

%% End of Script
