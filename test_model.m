% test_model.m
function [avg_loss, accuracy, precision, recall] = test_model(test_embs, Y_test, A, B, C_mat, Delta, W, b, epsilon)
% TEST_MODEL Evaluates the model on the test set.
%
% Inputs:
%   - test_embs: Cell array of test sequences, each cell contains a matrix of embeddings [D x T].
%   - Y_test: Cell array of one-hot encoded labels for test sequences [C x T].
%   - A, B, C_mat, Delta, W, b: Trained model parameters.
%   - epsilon: Small value for numerical stability.
%
% Outputs:
%   - avg_loss: Average cross-entropy loss over the test set.
%   - accuracy: Classification accuracy on the test set (in [0, 1]).
%   - precision: Precision for each class (1 x C).
%   - recall: Recall for each class (1 x C).

    % Initialize counters
    total_loss = 0;
    correct_predictions = 0;
    total_predictions = 0;
    num_classes = size(Y_test{1}, 1);  % Assuming Y_test is not empty
    confusion_matrix = zeros(num_classes, num_classes);  % Rows: true classes, Columns: predicted classes

    % Iterate over each test sample
    for j = 1:length(test_embs)
        % Extract the input sequence and labels
        X_seq = test_embs{j};    % Shape: [D x T]
        Y_seq = Y_test{j};       % Shape: [C x T]

        % Initialize hidden states
        N = size(A, 1);           % Number of latent states
        T_seq = size(X_seq, 2);   % Sequence length

        h = zeros(N, T_seq + 1);        % Hidden states [N x (T+1)]
        y_out = zeros(D, T_seq);        % Outputs [D x T]

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

        %% Forward Pass Through Time Steps
        for t_step = 1:T_seq
            x_t = X_seq(:, t_step);       % Input at time t [D x 1]
            h(:, t_step + 1) = A_d * h(:, t_step) + B_d * x_t;  % [N x 1]
            y_out(:, t_step) = C_mat * h(:, t_step + 1);        % [D x 1]
        end

        %% Output Layer Computation at Last Time Step (t = T)
        logits = W * y_out(:, T_seq) + b;       % [C x 1]

        % Compute softmax probabilities
        logits_stable = logits - max(logits);   % For numerical stability
        exp_logits = exp(logits_stable);
        sum_exp = sum(exp_logits);
        hat_y = exp_logits / (sum_exp + epsilon);  % [C x 1]

        % Compute cross-entropy loss
        y_true = Y_seq(:, T_seq);                   % [C x 1]
        loss = -sum(y_true .* log(hat_y + epsilon));
        total_loss = total_loss + loss;

        % Compute predicted and true classes
        [~, predicted_class] = max(hat_y);
        [~, true_class] = max(y_true);

        % Update confusion matrix
        confusion_matrix(true_class, predicted_class) = confusion_matrix(true_class, predicted_class) + 1;

        % Update accuracy
        if predicted_class == true_class
            correct_predictions = correct_predictions + 1;
        end
        total_predictions = total_predictions + 1;
    end

    % Compute average loss and accuracy
    avg_loss = total_loss / total_predictions;
    accuracy = correct_predictions / total_predictions;

    % Compute precision and recall per class
    precision = zeros(1, num_classes);
    recall = zeros(1, num_classes);

    for c = 1:num_classes
        tp = confusion_matrix(c, c);
        fp = sum(confusion_matrix(:, c)) - tp;
        fn = sum(confusion_matrix(c, :)) - tp;
        % Precision: TP / (TP + FP)
        if (tp + fp) > 0
            precision(c) = tp / (tp + fp);
        else
            precision(c) = 0;
        end
        % Recall: TP / (TP + FN)
        if (tp + fn) > 0
            recall(c) = tp / (tp + fn);
        else
            recall(c) = 0;
        end
    end
end
