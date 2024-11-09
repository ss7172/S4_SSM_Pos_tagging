% File: SSMLayer.m

classdef SSMLayer < nnet.layer.Layer
    % SSMLayer Custom layer implementing the SSM S4 model.
    properties (Learnable)
        A      % Continuous-time A matrix (D×N)
        B      % Continuous-time B matrix (D×N)
        C      % Continuous-time C matrix (D×N)
        Delta  % Discretization parameter (D×1)
    end

    methods
        function layer = SSMLayer(D, N, name)
            % Constructor for the SSMLayer
            % Inputs:
            %   - D: Embedding dimension
            %   - N: Hidden state dimension
            %   - name: Name of the layer

            % Set layer name
            layer.Name = name;

            % Initialize parameters
            layer.A = dlarray(-abs(randn(D, N)));    % Negative for stability
            layer.B = dlarray(randn(D, N));
            layer.C = dlarray(randn(D, N));
            layer.Delta = dlarray(0.1 * ones(D, 1)); % Initialize Delta (τΔ)
        end

        function [Z, memory] = forward(layer, X)
            % Forward pass of the SSM layer
            % Inputs:
            %   - X: Input data (B×L×D)
            % Outputs:
            %   - Z: Output data (B×L×D)
            %   - memory: Struct containing intermediate values for backpropagation

            [B, L, D] = size(X);
            N = size(layer.A, 2);

            % Discretize parameters
            [A_discrete, B_discrete] = discretize_parameters(layer.A, layer.B, layer.Delta);

            % Initialize hidden states
            h = dlarray(zeros(B, N, D, 'like', X));

            % Initialize output
            Z = dlarray(zeros(B, L, D, 'like', X));

            % Forward computation
            for t = 1:L
                x_t = X(:, t, :); % (B×1×D)
                x_t = permute(x_t, [1, 3, 2]); % (B×D×1)

                % Update hidden states
                h = h .* A_discrete + B_discrete .* x_t; % Element-wise operations

                % Compute output
                Z_t = sum(h .* layer.C, 2); % Sum over N dimension
                Z(:, t, :) = Z_t;
            end

            % Store memory for backward pass
            memory.X = X;
            memory.h = h;
            memory.A_discrete = A_discrete;
            memory.B_discrete = B_discrete;
        end

        function [dLdX, gradients] = backward(layer, X, ~, dLdZ, memory)
            % Backward pass of the SSM layer
            % Inputs:
            %   - X: Input data (B×L×D)
            %   - dLdZ: Gradient of the loss w.r.t. the output Z
            %   - memory: Struct containing intermediate values from forward pass
            % Outputs:
            %   - dLdX: Gradient of the loss w.r.t. the input X
            %   - gradients: Struct containing gradients w.r.t. learnable parameters

            % Since we are using automatic differentiation, we can leave this empty
            % MATLAB will handle gradients automatically for us
            dLdX = [];

            % Gradients w.r.t. learnable parameters are computed automatically
            gradients.A = [];
            gradients.B = [];
            gradients.C = [];
            gradients.Delta = [];
        end
    end
end

% Include the discretize_parameters function in this file or as a separate file.
