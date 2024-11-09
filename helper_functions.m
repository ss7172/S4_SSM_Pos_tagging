% File: helper_functions.m

% function [A_discrete, B_discrete] = discretize_parameters(A, B, Delta)
%     % DISCRETIZE_PARAMETERS Applies Zero-Order Hold (ZOH) discretization
%     %
%     % Inputs:
%     %   - A: Continuous-time A matrix (D×N)
%     %   - B: Continuous-time B matrix (D×N)
%     %   - Delta: Discretization parameter (D×1)
%     %
%     % Outputs:
%     %   - A_discrete: Discrete-time A matrix (D×N)
%     %   - B_discrete: Discrete-time B matrix (D×N)
% 
%     tau = Delta(1); % Assuming Delta is constant across D
%     [D, N] = size(A);
% 
%     A_discrete = exp(A * tau);
%     B_discrete = (A_discrete - eye(D, N)) / A * B * tau;
% end
