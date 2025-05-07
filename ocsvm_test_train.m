clc; clear; close all;
rng(42);
addpath(genpath('npy-matlab-master'));

% --- Setup ---
all_rooms = {'a', 'b', 'c'};
channels = {'channel_1', 'channel_36'};
base_folder = 'preprocessed_data';
occupied_label = -1;
k = 5;

% --- User input ---
fprintf('Available rooms: a, b, c\n');
train_room1 = input('Enter first training room: ', 's');
train_room2 = input('Enter second training room: ', 's');
train_rooms = {train_room1, train_room2};
test_room = setdiff(all_rooms, train_rooms);

fprintf('\n=== Training on Rooms: %s, %s | Testing on Room: %s ===\n', ...
    upper(train_rooms{1}), upper(train_rooms{2}), upper(test_room{1}));

% --- Load training data ---
X_train = [];
for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');
    for r = 1:length(train_rooms)
        filename = sprintf('preprocessed_%s_%s_empty.npy', train_rooms{r}, ch_num);
        filepath = fullfile(ch_folder, filename);
        if exist(filepath, 'file') == 2
            data = readNPY(filepath);
            [Nt, Btw, NrxNtx, Nf] = size(data);
            reshaped = reshape(data, Nt, Btw * NrxNtx * Nf);
            X_train = [X_train; reshaped];
        else
            warning('Missing: %s', filepath);
        end
    end
end
fprintf('Total training samples: %d\n', size(X_train, 1));

% --- Train OC-SVM ---
SVMModel = fitcsvm(X_train, ones(size(X_train, 1), 1), ...
    'KernelFunction', 'rbf', 'OutlierFraction', 0.0055, ...
    'BoxConstraint', 1, 'Standardize', true, 'KernelScale', 'auto');

% Save model
if ~exist('trained_models', 'dir'), mkdir('trained_models'); end
model_filename = sprintf('OCSVM_Model_%s%s.mat', train_rooms{1}, train_rooms{2});
save(fullfile('trained_models', model_filename), 'SVMModel');

% --- Load testing data ---
X_test = [];
y_test = [];
for ch = 1:length(channels)
    ch_folder = fullfile(base_folder, channels{ch});
    ch_num = extractAfter(channels{ch}, 'channel_');

    for type = ["empty", "occupied"]
        filename = sprintf('preprocessed_%s_%s_%s.npy', test_room{1}, ch_num, type);
        filepath = fullfile(ch_folder, filename);
        if exist(filepath, 'file') == 2
            data = readNPY(filepath);
            [Nt, Btw, NrxNtx, Nf] = size(data);
            reshaped = reshape(data, Nt, Btw * NrxNtx * Nf);
            X_test = [X_test; reshaped];
            if type == "empty"
                y_test = [y_test; ones(size(reshaped, 1), 1)];
            else
                y_test = [y_test; occupied_label * ones(size(reshaped, 1), 1)];
            end
        else
            warning('Missing: %s', filepath);
        end
    end
end
fprintf('Total testing samples: %d\n', length(y_test));

% --- Predict and Post-process ---
tic;
[~, scores] = predict(SVMModel, X_test);
y_pred = double(scores >= 0); y_pred(y_pred == 0) = -1;
y_pred_post = y_pred;
for i = k:length(y_pred)
    window = y_pred(i-k+1:i);
    y_pred_post(i) = sign(sum(window));
    if y_pred_post(i) == 0
        y_pred_post(i) = y_pred(i);
    end
end
inference_time = toc;

% --- Evaluation ---
accuracy_before = mean(y_test == y_pred);
accuracy_after = mean(y_test == y_pred_post);
fprintf('Accuracy Before Post-Processing: %.4f\n', accuracy_before);
fprintf('Accuracy After Post-Processing : %.4f\n', accuracy_after);
fprintf('Inference Time: %.4f sec | Per Sample: %.6f sec\n', ...
    inference_time, inference_time/length(y_test));

% --- Compute Evaluation Metrics (Before and After Post-Processing) ---
% Ground truth
y_true = y_test;
% Predictions
y_pred_raw = y_pred;
y_pred_smooth = y_pred_post;

% Confusion matrix components
[cm_raw, ~] = confusionmat(y_true, y_pred_raw);
[cm_post, ~] = confusionmat(y_true, y_pred_smooth);

metrics = @(cm) struct( ...
    'Accuracy', sum(diag(cm)) / sum(cm(:)), ...
    'Precision', cm(2,2) / sum(cm(:,2)), ...
    'Recall', cm(2,2) / sum(cm(2,:)), ...
    'F1', 2 * cm(2,2) / (2 * cm(2,2) + cm(1,2) + cm(2,1)) ...
);

scores_raw = metrics(cm_raw);
scores_post = metrics(cm_post);

%% --- PCA & OC-SVM Decision Boundary Visualization ---
[coeff, score_train] = pca(X_train);
X_train_pca = score_train(:, 1:2);
SVMModel_pca = fitcsvm(X_train_pca, ones(size(X_train_pca,1),1), ...
    'KernelFunction', 'rbf', 'OutlierFraction', 0.0055, ...
    'BoxConstraint', 1, 'Standardize', true, 'KernelScale', 'auto');

[x1Grid, x2Grid] = meshgrid(linspace(min(X_train_pca(:,1)), max(X_train_pca(:,1)), 100), ...
                            linspace(min(X_train_pca(:,2)), max(X_train_pca(:,2)), 100));
gridPoints = [x1Grid(:), x2Grid(:)];
[~, scores_grid] = predict(SVMModel_pca, gridPoints);

X_test_pca = (X_test - mean(X_train)) * coeff(:,1:2);
idx_empty = y_test == 1;
idx_occupied = y_test == -1;

figure;
hold on;
gscatter(X_train_pca(:,1), X_train_pca(:,2), ones(size(X_train_pca,1),1), 'b', '.', 10);
% contour(x1Grid, x2Grid, reshape(scores_grid, size(x1Grid)), [0 0], 'k');
contour(x1Grid, x2Grid, reshape(scores_grid, size(x1Grid)), [0 0], 'k-', 'LineWidth', 2);

plot(SVMModel_pca.SupportVectors(:,1), SVMModel_pca.SupportVectors(:,2), 'ro', 'MarkerSize', 10);
plot(X_test_pca(idx_empty,1), X_test_pca(idx_empty,2), 'bo', 'MarkerSize', 6, 'DisplayName', 'Test Empty');
plot(X_test_pca(idx_occupied,1), X_test_pca(idx_occupied,2), 'rx', 'MarkerSize', 6, 'DisplayName', 'Test Occupied');
legend({'Training Data (Empty)', 'Decision Boundary', 'Support Vectors', 'Test Empty', 'Test Occupied'}, 'Location', 'best');
title('OC-SVM: Training, Decision Boundary, and Test Samples (PCA)');
xlabel('Principal Component 1');
ylabel('Principal Component 2');
grid on;
hold off;

if ~exist('results', 'dir')
    mkdir('results');
end
saveas(gcf, fullfile('results', ...
    sprintf('ocsvm_pca_boundary_%s_%s_test_%s.png', ...
    train_rooms{1}, train_rooms{2}, test_room{1})));

% --- Save Confusion Matrices and Logs (OC-SVM style like CNN) ---
results_folder = 'results';
if ~exist(results_folder, 'dir')
    mkdir(results_folder);
end

% Convert labels to categorical
Y_test = categorical(y_test, [-1, 1], {'Occupied', 'Empty'});
Y_pred = categorical(y_pred, [-1, 1], {'Occupied', 'Empty'});
Y_pred_post = categorical(y_pred_post, [-1, 1], {'Occupied', 'Empty'});

% Confusion Matrix: Before Post-processing

fig1 = figure('Name', 'Confusion Matrix - Before Post-Processing');
confchart1 = confusionchart(Y_test, Y_pred, ...
    'Title', sprintf('Before Post Processing - Train [%s,%s] Test [%s]', ...
    train_rooms{1}, train_rooms{2}, test_room{1}), ...
    'Normalization', 'row-normalized');
set(gcf, 'Position', [100, 100, 600, 400]); % optional size adjustment
saveas(fig1, fullfile(results_folder, ...
    sprintf('ocsvm_confusion_before_%s_%s_test_%s.png', ...
    train_rooms{1}, train_rooms{2}, test_room{1})));

% --- Confusion Matrix: After Post-processing (Display + Save) ---
fig2 = figure('Name', 'Confusion Matrix - After Post-Processing');
confchart2 = confusionchart(Y_test, Y_pred_post, ...
    'Title', sprintf('After Post Processing - Train [%s,%s] Test [%s]', ...
    train_rooms{1}, train_rooms{2}, test_room{1}), ...
    'Normalization', 'row-normalized');
set(gcf, 'Position', [750, 100, 600, 400]); % optional size adjustment
saveas(fig2, fullfile(results_folder, ...
    sprintf('ocsvm_confusion_after_%s_%s_test_%s.png', ...
    train_rooms{1}, train_rooms{2}, test_room{1})));


% Plot Metrics Comparison
figure;
metric_names = {'Accuracy', 'Precision', 'Recall', 'F1'};
raw_vals = [scores_raw.Accuracy, scores_raw.Precision, scores_raw.Recall, scores_raw.F1];
post_vals = [scores_post.Accuracy, scores_post.Precision, scores_post.Recall, scores_post.F1];
b = bar([raw_vals; post_vals]' * 100);  % convert to percent
set(gca, 'XTickLabel', metric_names, 'FontSize', 12);
ylabel('Score (%)');
title(sprintf('Evaluation Metrics - Train [%s,%s] Test [%s]', ...
    train_rooms{1}, train_rooms{2}, test_room{1}));
grid on;

% Adjust bar colors (optional for clarity)
b(1).FaceColor = [0.2 0.6 0.8];  % Before
b(2).FaceColor = [0.85 0.33 0.1];  % After

% Move legend below the plot
legend({'Before Post-Processing', 'After Post-Processing'}, ...
    'Location', 'southoutside', 'Orientation', 'horizontal');

% Adjust figure size to fit legend
set(gcf, 'Position', [100, 100, 700, 500]);

% Save plot
saveas(gcf, fullfile(results_folder, ...
    sprintf('ocsvm_metrics_%s_%s_test_%s.png', ...
    train_rooms{1}, train_rooms{2}, test_room{1})));

%% --- Log file ---
% Estimate model size (in KB)
model_size_kb = numel(SVMModel.SupportVectors) * 8 / 1024;

log_filename = fullfile(results_folder, ...
    sprintf('ocsvm_log_%s_%s_test_%s.txt', ...
    train_rooms{1}, train_rooms{2}, test_room{1}));
fid = fopen(log_filename, 'w');

fprintf(fid, '=== OC-SVM Evaluation Log ===\n');
fprintf(fid, 'Train Rooms: %s, %s\n', upper(train_rooms{1}), upper(train_rooms{2}));
fprintf(fid, 'Test Room: %s\n', upper(test_room{1}));
fprintf(fid, 'Total Training Samples: %d\n', size(X_train, 1));
fprintf(fid, 'Total Testing Samples: %d\n', length(y_test));
fprintf(fid, 'OC-SVM Model Size (Support Vectors): %.2f KB\n', model_size_kb);
fprintf(fid, 'Raw Accuracy: %.2f%%\n', accuracy_before * 100);
fprintf(fid, 'Post-Processed Accuracy (k=%d): %.2f%%\n', k, accuracy_after * 100);
fprintf(fid, 'Inference Time (Total): %.4f seconds\n', inference_time);
fprintf(fid, 'Average Inference Time per Sample: %.6f seconds\n', inference_time / length(y_test));
fprintf(fid, '\n--- Detailed Metrics ---\n');
fprintf(fid, 'Before Post-Processing:\n');
fprintf(fid, '  Accuracy : %.2f%%\n', scores_raw.Accuracy * 100);
fprintf(fid, '  Precision: %.2f%%\n', scores_raw.Precision * 100);
fprintf(fid, '  Recall   : %.2f%%\n', scores_raw.Recall * 100);
fprintf(fid, '  F1 Score : %.2f%%\n', scores_raw.F1 * 100);

fprintf(fid, '\nAfter Post-Processing:\n');
fprintf(fid, '  Accuracy : %.2f%%\n', scores_post.Accuracy * 100);
fprintf(fid, '  Precision: %.2f%%\n', scores_post.Precision * 100);
fprintf(fid, '  Recall   : %.2f%%\n', scores_post.Recall * 100);
fprintf(fid, '  F1 Score : %.2f%%\n', scores_post.F1 * 100);

fclose(fid);



