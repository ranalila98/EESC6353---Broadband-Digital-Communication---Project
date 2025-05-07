clc; clear; close all;
rng(42);
addpath(genpath('npy-matlab-master'));

% === User Input ===
all_rooms = {'a', 'b', 'c'};
channels = {'channel_1', 'channel_36'};
classes = {'empty', 'occupied'};
base_folder = 'preprocessed_data';
k = 7;

fprintf('Available rooms: a, b, c\n');
train_room1 = input('Enter first training room (e.g., a): ', 's');
train_room2 = input('Enter second training room (e.g., b): ', 's');
train_rooms = {train_room1, train_room2};
test_room = setdiff(all_rooms, train_rooms);

% === TRAINING ===
X = {};
Y = {};
count = containers.Map(classes, {0, 0});

for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');

    for r = 1:length(train_rooms)
        for cls = 1:length(classes)
            filename = sprintf('preprocessed_%s_%s_%s.npy', train_rooms{r}, ch_num, classes{cls});
            filepath = fullfile(ch_folder, filename);

            if exist(filepath, 'file') == 2
                data = readNPY(filepath);
                [Nt, Btw, NrxNtx, Nf] = size(data);
                reshaped = reshape(data, Nt, []);
                for ii = 1:Nt
                    X{end+1} = reshaped(ii, :)';
                    Y{end+1} = classes{cls};
                end
                count(classes{cls}) = count(classes{cls}) + Nt;
            else
                warning('Missing: %s', filepath);
            end
        end
    end
end

fprintf('\n=== Training on Rooms: %s, %s ===\n', upper(train_rooms{1}), upper(train_rooms{2}));
fprintf('Total training samples: %d\n', numel(Y));
disp('Class distribution:');
disp(count);

Y = categorical(string(Y));
X = cat(2, X{:});
X = num2cell(X, 1);
numFeatures = size(X{1}, 1);
numClasses = numel(categories(Y));
classWeights = 1 ./ cell2mat(values(count));
classWeights = classWeights / sum(classWeights);

layers = [
    sequenceInputLayer(numFeatures, 'MinLength', 1)
    convolution1dLayer(5, 16, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    convolution1dLayer(3, 32, 'Padding', 'same')
    batchNormalizationLayer
    reluLayer
    dropoutLayer(0.2)
    globalAveragePooling1dLayer
    fullyConnectedLayer(numClasses)
    softmaxLayer
    classificationLayer('Classes', categories(Y), 'ClassWeights', classWeights)
];

options = trainingOptions('adam', ...
    'MaxEpochs', 20, ...
    'MiniBatchSize', 32, ...
    'InitialLearnRate', 1e-3, ...
    'Shuffle', 'every-epoch', ...
    'Verbose', false, ...
    'Plots', 'training-progress');

net = trainNetwork(X, Y, layers, options);
% Calculate CNN size in KB, ensuring only layers with Weights and Bias are considered
totalParams = 0;
for i = 1:numel(net.Layers)
    layer = net.Layers(i);
    if isprop(layer, 'Weights') && ~isempty(layer.Weights)
        totalParams = totalParams + numel(layer.Weights);
    end
    if isprop(layer, 'Bias') && ~isempty(layer.Bias)
        totalParams = totalParams + numel(layer.Bias);
    end
end
cnn_size = totalParams * 4 / 1024;  % Assuming 4 bytes per parameter
fprintf('CNN size: %.2f KB\n', cnn_size);

analyzeNetwork(net); 
% % CNN size in KB
% totalParams = sum(arrayfun(@(l) numel(l.Weights) + numel(l.Bias), net.Layers, 'UniformOutput', true));
% cnn_size = totalParams * 4 / 1024;
% fprintf('CNN size: %.2f KB\n', cnn_size);
% view(net)
% deepNetworkDesigner(net)

% Save training progress figure
if ~exist('plots', 'dir')
    mkdir('plots');
end

figHandles = findall(groot, 'Type', 'Figure', 'Name', 'Training Progress');
if ~isempty(figHandles)
    saveas(figHandles(1), fullfile('plots', ...
        sprintf('training_progress_%s_%s_test_%s.png', ...
        train_rooms{1}, train_rooms{2}, test_room{1})));
    close(figHandles(1));  % Close the figure after saving
end

% === Save Trained Model ===
if ~exist('trained_models', 'dir')
    mkdir('trained_models');
end

model_filename = sprintf('CNN1D_Model_%s%s.mat', train_rooms{1}, train_rooms{2});
save(fullfile('trained_models', model_filename), 'net');
fprintf('Saved trained model: %s\n', model_filename);

% === TESTING ===
X_test = {};
Y_test = {};

for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');

    for cls = 1:length(classes)
        filename = sprintf('preprocessed_%s_%s_%s.npy', test_room{1}, ch_num, classes{cls});
        filepath = fullfile(ch_folder, filename);

        if exist(filepath, 'file') == 2
            data = readNPY(filepath);
            [Nt, Btw, NrxNtx, Nf] = size(data);
            reshaped = reshape(data, Nt, []);
            for ii = 1:Nt
                X_test{end+1} = reshaped(ii, :)';
                Y_test{end+1} = classes{cls};
            end
        else
            warning('Missing: %s', filepath);
        end
    end
end

Y_test = categorical(string(Y_test));
tic;
% Y_pred = classify(net, X_test);
% training_time = toc;
% fprintf('Training Time: %.2f seconds\n', training_time);

tic;
Y_pred = classify(net, X_test);


% === Majority Voting ===
Y_pred_post = Y_pred;
for j = k:length(Y_pred)
    window = Y_pred(j-k+1:j);
    Y_pred_post(j) = mode(window);
end

inference_time = toc;
fprintf('Inference Time (Total): %.4f seconds\n', inference_time);
fprintf('Average Inference Time per Sample: %.6f seconds\n', inference_time / numel(Y_test));

% === Evaluation ===
[cm_raw, order] = confusionmat(Y_test, Y_pred);
[cm_post, ~] = confusionmat(Y_test, Y_pred_post);

acc_raw = mean(diag(cm_raw ./ sum(cm_raw, 2)));
acc_post = mean(diag(cm_post ./ sum(cm_post, 2)));

cm_raw_pct = cm_raw ./ sum(cm_raw, 2) * 100;
cm_post_pct = cm_post ./ sum(cm_post, 2) * 100;

fprintf('\n=== Testing on Room: %s ===\n', upper(test_room{1}));
fprintf('Raw Accuracy: %.2f%%\n', acc_raw * 100);
disp(array2table(cm_raw_pct, 'VariableNames', cellstr(order), 'RowNames', cellstr(order)));

fprintf('Post-Processed Accuracy (k = %d): %.2f%%\n', k, acc_post * 100);
disp(array2table(cm_post_pct, 'VariableNames', cellstr(order), 'RowNames', cellstr(order)));


% --- Plot and Save Confusion Matrices ---
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Raw confusion matrix
fig1 = figure('Visible', 'off');
confusionchart(Y_test, Y_pred, ...
    'Title', sprintf('Before Post Processing - Train [%s,%s] Test [%s]', train_rooms{1}, train_rooms{2}, test_room{1}), ...
    'Normalization', 'row-normalized');
saveas(fig1, fullfile(results_folder, ...
    sprintf('cnn_confusion_before_%s_%s_test_%s.png', train_rooms{1}, train_rooms{2}, test_room{1})));
close(fig1);

% Post-processed confusion matrix
fig2 = figure('Visible', 'off');
confusionchart(Y_test, Y_pred_post, ...
    'Title', sprintf('After Post Processing - Train [%s,%s] Test [%s]', train_rooms{1}, train_rooms{2}, test_room{1}), ...
    'Normalization', 'row-normalized');
saveas(fig2, fullfile(results_folder, ...
    sprintf('cnn_confusion_after_%s_%s_test_%s.png', train_rooms{1}, train_rooms{2}, test_room{1})));
close(fig2);

log_filename = fullfile(results_folder, ...
    sprintf('cnn_log_%s_%s_test_%s.txt', train_rooms{1}, train_rooms{2}, test_room{1}));
fid = fopen(log_filename, 'w');

fprintf(fid, '=== CNN Evaluation Log ===\n');
fprintf(fid, 'Train Rooms: %s, %s\n', upper(train_rooms{1}), upper(train_rooms{2}));
fprintf(fid, 'Test Room: %s\n', upper(test_room{1}));
fprintf(fid, 'Total Training Samples: %d\n', numel(Y));
fprintf(fid, 'Total Testing Samples: %d\n', numel(Y_test));
fprintf(fid, 'CNN Model Size: %.2f KB\n', cnn_size);
fprintf(fid, 'Raw Accuracy: %.2f%%\n', acc_raw * 100);
fprintf(fid, 'Post-Processed Accuracy (k=%d): %.2f%%\n', k, acc_post * 100);
fprintf(fid, 'Inference Time (Total): %.4f seconds\n', inference_time);
fprintf(fid, 'Average Inference Time per Sample: %.6f seconds\n', inference_time / numel(Y_test));


fclose(fid);
fprintf('Saved evaluation log: %s\n', log_filename);
% % === Ask for Another Run ===
% user_continue = input('\nDo you want to train and test with another room combination? (y/n): ', 's');
% if strcmpi(user_continue, 'y')
%     run(mfilename);  % Rerun the current script
% else
%     fprintf('Exiting... All evaluations completed.\n');
% end
