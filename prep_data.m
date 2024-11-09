function [train_embs, test_embs, valid_embs, Y_train, Y_valid, Y_test] = prep_data(train_data, valid_data, test_data, word2embedding)
    % PREP_DATA Preprocesses training, validation, and test data for SSMs.
    %
    % [train_embs, test_embs, valid_embs, Y_train, Y_valid, Y_test] = prep_data(train_data, valid_data, test_data, word2embedding)
    %
    % Inputs:
    %   - train_data: Struct containing training data with fields 'tokens' and 'pos_tags'.
    %   - valid_data: Struct containing validation data with fields 'tokens' and 'pos_tags'.
    %   - test_data: Struct containing test data with fields 'tokens' and 'pos_tags'.
    %   - word2embedding: containers.Map object mapping words to their embedding vectors.
    %
    % Outputs:
    %   - train_embs: Cell array of embedded training sequences.
    %   - test_embs: Cell array of embedded test sequences.
    %   - valid_embs: Cell array of embedded validation sequences.
    %   - Y_train: Cell array of one-hot encoded training labels.
    %   - Y_valid: Cell array of one-hot encoded validation labels.
    %   - Y_test: Cell array of one-hot encoded test labels.

    % Embed tokens and retrieve labels
    [train_embs, train_labels] = embed(train_data, word2embedding, 'train');
    [valid_embs, valid_labels] = embed(valid_data, word2embedding, 'valid');
    [test_embs, test_labels] = embed(test_data, word2embedding, 'test');
    
    % Define the fixed sequence length for chunking
    SEQ_LEN = 4;
    
    % Convert string labels to numeric
    train_labels = cellfun(@(str_labels) str2num(str_labels), train_labels, 'UniformOutput', false);
    valid_labels = cellfun(@(str_labels) str2num(str_labels), valid_labels, 'UniformOutput', false);
    test_labels = cellfun(@(str_labels) str2num(str_labels), test_labels, 'UniformOutput', false);
    
    % Simplify tags into broader categories
    train_labels = convert_tags(train_labels);
    valid_labels = convert_tags(valid_labels);
    test_labels = convert_tags(test_labels);

    % One-hot encode the simplified labels
    [Y_train, Y_valid, Y_test] = one_hot(train_labels, valid_labels, test_labels, SEQ_LEN);

    % Implement Overlapping Sliding Window Chunking
    [train_embs, Y_train] = chunk_data(train_embs, Y_train, SEQ_LEN);
    [valid_embs, Y_valid] = chunk_data(valid_embs, Y_valid, SEQ_LEN);
    [test_embs, Y_test] = chunk_data(test_embs, Y_test, SEQ_LEN);
end

function [Y_train, Y_valid, Y_test] = one_hot(train_labels, valid_labels, test_labels, SEQ_LEN)
    % ONE_HOT Converts simplified POS tags into one-hot encoded vectors.
    %
    % [Y_train, Y_valid, Y_test] = one_hot(train_labels, valid_labels, test_labels, SEQ_LEN)
    %
    % Inputs:
    %   - train_labels: Cell array of numeric training labels.
    %   - valid_labels: Cell array of numeric validation labels.
    %   - test_labels: Cell array of numeric test labels.
    %   - SEQ_LEN: Fixed sequence length for one-hot encoding.
    %
    % Outputs:
    %   - Y_train: Cell array of one-hot encoded training labels.
    %   - Y_valid: Cell array of one-hot encoded validation labels.
    %   - Y_test: Cell array of one-hot encoded test labels.

    % Initialize cell arrays for one-hot encoded labels
    Y_train = cell(size(train_labels));
    Y_valid = cell(size(valid_labels));
    Y_test = cell(size(test_labels));
    
    % Number of classes (noun, verb, adj/adv, other)
    num_classes = 4;
    
    % One-hot encode training labels
    for i = 1:length(train_labels)
        sample_labels = zeros(SEQ_LEN, num_classes);
        for j = 1:length(train_labels{i})
            if train_labels{i}(j) >=1 && train_labels{i}(j) <= num_classes
                sample_labels(j, train_labels{i}(j)) = 1;
            end
        end
        Y_train{i} = sample_labels;
    end
    
    % One-hot encode validation labels
    for i = 1:length(valid_labels)
        sample_labels = zeros(SEQ_LEN, num_classes);
        for j = 1:length(valid_labels{i})
            if valid_labels{i}(j) >=1 && valid_labels{i}(j) <= num_classes
                sample_labels(j, valid_labels{i}(j)) = 1;
            end
        end
        Y_valid{i} = sample_labels;
    end
    
    % One-hot encode test labels
    for i = 1:length(test_labels)
        sample_labels = zeros(SEQ_LEN, num_classes);
        for j = 1:length(test_labels{i})
            if test_labels{i}(j) >=1 && test_labels{i}(j) <= num_classes
                sample_labels(j, test_labels{i}(j)) = 1;
            end
        end
        Y_test{i} = sample_labels;
    end
end

function new_tags = convert_tags(old_tags)
    % CONVERT_TAGS Simplifies POS tags into broader categories.
    %
    % new_tags = convert_tags(old_tags)
    %
    % Inputs:
    %   - old_tags: Cell array of numeric original POS tags.
    %
    % Outputs:
    %   - new_tags: Cell array of simplified numeric POS tags.
    %
    % Tag Mapping:
    %   - Nouns: 1
    %   - Verbs: 2
    %   - Adjectives/Adverbs: 3
    %   - Others: 4

    % Define tag categories
    noun_tags = [21, 24, 22, 23, 25, 28, 29];
    verb_tags = [37, 38, 39, 40, 41, 42];
    adj_adv_tags = [16, 17, 18, 30, 31, 32];
    % Default is 4 (other)
    % noun (1), verb (2), adjective/adverb (3)
    
    % Initialize the new_tags cell array
    new_tags = cell(size(old_tags));
    for i = 1:length(old_tags)
        old_tags_i = old_tags{i};
        new_tags_i = 4 * ones(1, length(old_tags_i)); % Default to 'Other'
    
        % Assign new category indices
        new_tags_i(ismember(old_tags_i, noun_tags)) = 1;
        new_tags_i(ismember(old_tags_i, verb_tags)) = 2;
        new_tags_i(ismember(old_tags_i, adj_adv_tags)) = 3;
    
        new_tags{i} = new_tags_i;
    end
end

function [unpadded_embs, labels] = embed(data, word2embedding, split)
    % EMBED Converts words to their corresponding embedding vectors.
    %
    % [unpadded_embs, labels] = embed(data, word2embedding, split)
    %
    % Inputs:
    %   - data: Struct containing data with fields 'tokens' and 'pos_tags'.
    %   - word2embedding: containers.Map object mapping words to their embedding vectors.
    %   - split: String indicating the data split ('train', 'valid', 'test').
    %
    % Outputs:
    %   - unpadded_embs: Cell array of embedded sequences.
    %   - labels: Cell array of POS tags corresponding to the sequences.

    fprintf('Embedding %s... \n', split);
    unpadded_embs = cell(size(data.tokens));
    valid_samples = true(length(unpadded_embs), 1); % All samples are considered valid
    n_invalid = 0;
    
    for i = 1:length(unpadded_embs)
        tokens = data.tokens{i};
        % Extract words by removing surrounding brackets and splitting by comma
        words = cellfun(@(x) x(2:end-1), strsplit(tokens(2:end-1), ', '), 'UniformOutput', false);
        
        sentence_emb = cell(size(words));
        for j = 1:length(words)
            word = words{j};
            word = lower(word); % Convert word to lowercase
        
            if isKey(word2embedding, word)
                sentence_emb{j} = word2embedding(word);
            else
                % Assign zero vector for unknown words
                sentence_emb{j} = zeros(1,64);
                n_invalid = n_invalid + 1;
            end
        end
        
        if valid_samples(i)
            % Concatenate embeddings vertically
            unpadded_embs{i} = cell2mat(sentence_emb');
        end
    end
    
    fprintf('Found %d unknown embeddings \n', n_invalid);
    old_len = length(unpadded_embs);
    unpadded_embs = unpadded_embs(valid_samples);
    fprintf('Removed %d / %d samples \n', old_len - length(unpadded_embs), old_len);
    labels = data.pos_tags(valid_samples);
end

function [X_chunked, Y_chunked] = chunk_data(X_old, Y_old, chunk_size)
    % CHUNK_DATA Creates overlapping sliding window chunks from sequences.
    %
    % [X_chunked, Y_chunked] = chunk_data(X_old, Y_old, chunk_size)
    %
    % Inputs:
    %   - X_old: Cell array of embedded sequences.
    %   - Y_old: Cell array of one-hot encoded labels.
    %   - chunk_size: Size of each chunk (e.g., 4).
    %
    % Outputs:
    %   - X_chunked: Cell array of chunked embedded sequences.
    %   - Y_chunked: Cell array of chunked one-hot encoded labels.

    % Initialize empty cell arrays for chunked data
    X_chunked = {};
    Y_chunked = {};
    
    for i = 1:length(X_old)
        sentence = X_old{i};
        labels = Y_old{i};
    
        num_tokens = size(sentence, 1);
        if num_tokens >= chunk_size
            % Calculate the number of overlapping chunks
            num_chunks = num_tokens - chunk_size + 1;
    
            for j = 1:num_chunks
                start_idx = j;
                end_idx = j + chunk_size - 1;
    
                % Extract the chunk from embeddings and labels
                tokens_chunk = sentence(start_idx:end_idx, :);
                labels_chunk = labels(start_idx:end_idx, :);
    
                % Append the chunk to the output cell arrays
                X_chunked{end+1, 1} = tokens_chunk;
                Y_chunked{end+1, 1} = labels_chunk;
            end
        end
    end
end
