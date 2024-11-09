function [X, Y] = prepare_data(embs, labels)
    % PREPARE_DATA Converts cell arrays to tensors and categorical labels.
    
    numObservations = numel(embs);
    X = cell(numObservations, 1);
    Y = cell(numObservations, 1);
    
    for i = 1:numObservations
        X{i} = embs{i}'; % Transpose to D×L for sequenceInputLayer
        % Convert one-hot labels to categorical
        [~, labelIndices] = max(labels{i}, [], 2);
        Y{i} = categorical(labelIndices);
    end
end
