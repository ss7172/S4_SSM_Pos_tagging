% train_s4.m
% Training Script for the SSM Model with Hyperparameter Tuning using Validation Set

% Clear workspace and command window
clear;
clc;

%% Load and Prepare Data
% Ensure 'prep_data.m' is in your MATLAB path or current directory

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

% Ensure "nan" embedding exists
if ~isKey(word2embedding, 'nan')
    error('"nan" embedding not found in the embeddings map.');
end

% Prepare data using 'prep_data.m' script
% [train_embs, test_embs, valid_embs, Y_train, Y_test, Y_valid] = prep_data(...);
% Ensure that 'prep_data.m' maps OOV words to "nan" embedding and returns:
% - train_embs: Cell array, each cell [D x T]
% - valid_embs: Cell array, each cell [D x T]
% - test_embs: Cell array, each cell [D x T]
% - Y_train, Y_valid, Y_test: Cell arrays, each cell [C x T]
[train_embs, test_embs, valid_embs, Y_train, Y_test, Y_valid] = prep_data(train_data, valid_data, test_data, word2embedding);

%% Set Hyperparameters for Tuning
learning_rates = [0.0001, 0.00005, 0.00001];
regularization_strengths = [1e-3, 1e-4, 1e-5];
best_val_accuracy = 0;
best_hyperparams = struct('eta', 0, 'lambda', 0);
epsilon = 1e-6;         % Small value for numerical stability
num_epochs = 10;        % Number of training epochs
max_grad_norm = 1.0;    % Maximum gradient norm for clipping
patience_limit = 3;     % Patience for early stopping
N = 8;                  % Number of latent states
D = 64;                 % Embedding dimension
C = 4;                  % Number of classes

%% Hyperparameter Tuning Loop
for lr_idx = 1:length(learning_rates)
    for lambda_idx = 1:length(regularization_strengths)
        % Set hyperparameters
        eta_initial = learning_rates(lr_idx);
        eta = eta_initial;
        lambda = regularization_strengths(lambda_idx);
        fprintf('\nTesting Learning Rate: %.5f, Regularization Strength: %.1e\n', eta, lambda);

        %% Initialize Model Parameters with Xavier Initialization
        rng('default');  % For reproducibility

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
        b = zeros(C, 1);  % Bias vector initialized to zero

        % Initialize Delta with positive values
        Delta = abs(randn(N, 1)) + 0.1;  % Ensure Delta has values >= 0.1

        %% Initialize Logging Variables
        training_losses = zeros(num_epochs, 1);
        validation_losses = zeros(num_epochs, 1);
        validation_accuracies = zeros(num_epochs, 1);
        training_gradient_norms = zeros(num_epochs, 1);
        patience_counter = 0;
        best_val_loss = Inf;

        %% Training Loop
        num_samples = length(train_embs);

        for epoch = 1:num_epochs
            fprintf('Epoch %d/%d\n', epoch, num_epochs);

            epoch_loss = 0;

            for i = 1:num_samples
                % Extract the input sequence and labels
                X_seq = train_embs{i};    % Shape: [D x T]
                Y_seq = Y_train{i};       % Shape: [C x T]

                % Ensure sequence length is sufficient
                if size(X_seq, 2) < 4
                    continue;  % Skip samples with insufficient length
                end

                % Initialize hidden states
                h = zeros(N, size(X_seq, 2) + 1);  % [N x (T+1)]
                y_out = zeros(D, size(X_seq, 2));  % [D x T]

                %% Discretization Step
                D_mat = diag(Delta) * A;  % [N x N]
                I_N = eye(N);

                % Regularization for numerical stability
                D_mat_reg = D_mat + epsilon * I_N;

                % Compute matrix exponential
                A_d = expm(D_mat);  % [N x N]

                % Solve for B_d
                RHS = (A_d - I_N) * (diag(Delta) * B);  % [N x D]
                B_d = D_mat_reg \ RHS;  % [N x D]

                %% Dimension Checks (Debugging)
                assert(size(A_d, 1) == N && size(A_d, 2) == N, ...
                    sprintf('A_d dimensions incorrect at sample %d, epoch %d. Expected [%d x %d], got [%d x %d].', ...
                    i, epoch, N, N, size(A_d, 1), size(A_d, 2)));
                assert(size(B_d, 1) == N && size(B_d, 2) == D, ...
                    sprintf('B_d dimensions incorrect at sample %d, epoch %d. Expected [%d x %d], got [%d x %d].', ...
                    i, epoch, N, D, size(B_d, 1), size(B_d, 2)));
                assert(size(X_seq, 1) == D, ...
                    sprintf('X_seq dimensions incorrect at sample %d, epoch %d. Expected [%d x %d], got [%d x %d].', ...
                    i, epoch, D, size(X_seq, 2), size(X_seq, 1), size(X_seq, 2)));

                %% Forward Pass Through Time Steps
                T_seq = size(X_seq, 2);  % Sequence length (should be 4)
                for t_step = 1:T_seq
                    x_t = X_seq(:, t_step);       % Input at time t [D x 1]

                    % Dimension check for x_t
                    assert(size(x_t, 1) == D && size(x_t, 2) == 1, ...
                        sprintf('x_t dimensions incorrect at sample %d, epoch %d, t_step %d. Expected [%d x 1], got [%d x %d].', ...
                        i, epoch, t_step, D, size(x_t, 1), size(x_t, 2)));

                    % Compute hidden state
                    h(:, t_step + 1) = A_d * h(:, t_step) + B_d * x_t;  % [N x 1]

                    % Compute output
                    y_out(:, t_step) = C_mat * h(:, t_step + 1);        % [D x 1]
                end

                %% Output Layer Computation at Last Time Step
                logits = W * y_out(:, T_seq) + b;       % [C x 1]

                % Compute softmax probabilities
                logits_stable = logits - max(logits);   % For numerical stability
                exp_logits = exp(logits_stable);
                sum_exp = sum(exp_logits);
                hat_y = exp_logits / (sum_exp + epsilon);  % [C x 1]

                % Compute cross-entropy loss
                y_true = Y_seq(:, T_seq);                   % [C x 1]
                loss = -sum(y_true .* log(hat_y + epsilon));
                epoch_loss = epoch_loss + loss;

                %% Backward Pass
                % Gradient w.r.t logits
                dL_dlogits = hat_y - y_true;          % [C x 1]

                % Gradients w.r.t W and b
                dL_dW = dL_dlogits * y_out(:, T_seq)';    % [C x D]
                dL_db = dL_dlogits;                       % [C x 1]

                % Gradient w.r.t y_T
                dL_dy_T = W' * dL_dlogits;             % [D x 1]

                % Gradient w.r.t C_mat and h_T
                dL_dC = dL_dy_T * h(:, T_seq + 1)';        % [D x N]
                dL_dh = zeros(N, T_seq + 1);              % [N x (T+1)]
                dL_dh(:, T_seq + 1) = C_mat' * dL_dy_T;    % [N x 1]

                % Backpropagation through time
                dL_dA_d = zeros(N, N);
                dL_dB_d = zeros(N, D);

                for t_step = T_seq:-1:1
                    % Gradients w.r.t A_d and B_d
                    dL_dA_d = dL_dA_d + dL_dh(:, t_step + 1) * h(:, t_step)';  % [N x N]
                    dL_dB_d = dL_dB_d + dL_dh(:, t_step + 1) * X_seq(:, t_step)';  % [N x D]

                    % Update gradient w.r.t h_t
                    dL_dh(:, t_step) = A_d' * dL_dh(:, t_step + 1);  % [N x 1]
                end

                %% Gradients w.r.t Original Parameters
                % Gradient w.r.t A
                dA_dA = diag(Delta) * A_d;  % [N x N]
                dL_dA = dL_dA_d .* dA_dA;    % Element-wise multiplication

                % Gradient w.r.t B
                dB_dB = diag(Delta);         % [N x N]
                dL_dB = dL_dB_d .* (dB_dB * ones(1, D));  % [N x D]

                % Gradient w.r.t Delta
                dL_dDelta = zeros(N, 1);
                for i_delta = 1:N
                    % Compute partial derivatives
                    dA_dDelta_i = A(:, i_delta) * A_d(i_delta, :)' .* Delta(i_delta);  % [N x 1]
                    dB_dDelta_i = B(:, i_delta) * Delta(i_delta);                      % [D x 1]

                    % Accumulate gradients
                    dL_dDelta(i_delta) = sum(dL_dA_d(i_delta, :) .* dA_dDelta_i') + ...
                                          sum(dL_dB_d(i_delta, :) .* dB_dDelta_i');
                end

                %% Gradient Clipping
                % Compute total norm of gradients
                gradient_list = {dL_dA, dL_dB, dL_dC, dL_dDelta, dL_dW, dL_db};
                total_norm = 0;
                for grad = 1:length(gradient_list)
                    total_norm = total_norm + sum(gradient_list{grad}{1}(:).^2);
                end
                total_norm = sqrt(total_norm);
                training_gradient_norms(epoch) = total_norm;

                % Compute scaling factor
                scaling_factor = min(1, max_grad_norm / (total_norm + epsilon));

                % Scale gradients
                dL_dA = dL_dA * scaling_factor;
                dL_dB = dL_dB * scaling_factor;
                dL_dC = dL_dC * scaling_factor;
                dL_dDelta = dL_dDelta * scaling_factor;
                dL_dW = dL_dW * scaling_factor;
                dL_db = dL_db * scaling_factor;

                %% Apply Regularization (Weight Decay)
                dL_dA = dL_dA + lambda * A;
                dL_dB = dL_dB + lambda * B;
                dL_dC = dL_dC + lambda * C_mat;
                dL_dW = dL_dW + lambda * W;
                % Biases are typically not regularized

                %% Parameter Updates
                A = A - eta * dL_dA;
                B = B - eta * dL_dB;
                C_mat = C_mat - eta * dL_dC;
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

                %% Logging
                training_losses(epoch) = training_losses(epoch) + loss;
            end

            % Compute average loss for the epoch
            avg_epoch_loss = training_losses(epoch) / num_samples;
            fprintf('Epoch %d completed. Average Training Loss: %.4f\n', epoch, avg_epoch_loss);

            %% Validation Evaluation
            [validation_loss, validation_accuracy, validation_precision, validation_recall] = ...
                validate_model(valid_embs, Y_valid, A, B, C_mat, Delta, W, b, epsilon);
            validation_losses(epoch) = validation_loss;
            validation_accuracies(epoch) = validation_accuracy;
            fprintf('Validation Loss: %.4f, Validation Accuracy: %.2f%%\n', validation_loss, validation_accuracy * 100);

            % Display Precision and Recall per class
            for c = 1:length(validation_precision)
                fprintf('Class %d - Precision: %.2f%%, Recall: %.2f%%\n', ...
                    c, validation_precision(c) * 100, validation_recall(c) * 100);
            end

            %% Early Stopping Implementation
            if validation_accuracy > best_val_accuracy
                best_val_loss = validation_loss;
                best_val_accuracy = validation_accuracy;
                best_hyperparams.eta = eta_initial;
                best_hyperparams.lambda = lambda;
                save('best_trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');
                patience_counter = 0;
                fprintf('New best model found. Saving model.\n');
            else
                patience_counter = patience_counter + 1;
                if patience_counter >= patience_limit
                    fprintf('Early stopping triggered at epoch %d.\n', epoch);
                    break;
                end
            end

            %% Learning Rate Scheduling (Optional)
            % Example: Exponential Decay
            eta = eta_initial * exp(-0.05 * epoch);
        end

        % Reset patience counter for the next hyperparameter set
        patience_counter = 0;

        fprintf('\nBest Hyperparameters:\n');
        fprintf('Learning Rate: %.5f\n', best_hyperparams.eta);
        fprintf('Regularization Strength: %.1e\n', best_hyperparams.lambda);
        fprintf('Best Validation Accuracy: %.2f%%\n', best_val_accuracy * 100);
    end
end


    %% Testing Evaluation
    % Load the best trained model parameters
    load('best_trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');

    % Evaluate on the test set
    [test_loss, test_accuracy, test_precision, test_recall] = ...
        test_model(test_embs, Y_test, A, B, C_mat, Delta, W, b, epsilon);
    fprintf('\nTest Loss: %.4f, Test Accuracy: %.2f%%\n', test_loss, test_accuracy * 100);

    % Display Precision and Recall per class
    for c = 1:length(test_precision)
        fprintf('Class %d - Precision: %.2f%%, Recall: %.2f%%\n', ...
            c, test_precision(c) * 100, test_recall(c) * 100);
    end

    %% Save the Trained Model Parameters
    save('trained_model.mat', 'A', 'B', 'C_mat', 'Delta', 'W', 'b');

    %% Plotting Training and Validation Losses
    figure;
    plot(1:num_epochs, training_losses(1:num_epochs), '-o', 'DisplayName', 'Training Loss');
    hold on;
    plot(1:num_epochs, validation_losses(1:num_epochs), '-x', 'DisplayName', 'Validation Loss');
    xlabel('Epoch');
    ylabel('Loss');
    title('Training and Validation Loss Over Epochs');
    legend('show');
    grid on;

    %% Plotting Gradient Norms
    figure;
    plot(1:num_epochs, training_gradient_norms(1:num_epochs), '-s');
    xlabel('Epoch');
    ylabel('Gradient Norm');
    title('Gradient Norm Over Epochs');
    grid on;

    %% End of Script
