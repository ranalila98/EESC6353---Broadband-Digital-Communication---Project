function plot_metrics_summary(metrics_before, metrics_after, unique_channels, unique_rooms)
    fields = {'Accuracy', 'Precision', 'Recall', 'F1_Score'};

    % Create results folder if it doesn't exist
    results_dir = 'results';
    if ~exist(results_dir, 'dir')
        mkdir(results_dir);
    end

    %% Channel-wise Metrics with single legend
    num_channels = length(unique_channels);
    fig1 = figure;
    tl = tiledlayout(1, num_channels, 'TileSpacing', 'compact', 'Padding', 'compact');
    for ch = 1:num_channels
        nexttile;
        data_before = zeros(1, length(fields));
        data_after = zeros(1, length(fields));
        for i = 1:length(fields)
            data_before(i) = metrics_before.(unique_channels{ch}).(fields{i});
            data_after(i) = metrics_after.(unique_channels{ch}).(fields{i});
        end
        bar([data_before' data_after']);
        title(sprintf('Channel: %s', unique_channels{ch}));
        ylabel('Score');
        ylim([0 1]);
        xticks(1:length(fields));
        xticklabels(fields);
        xtickangle(45);
        grid on;
    end
    lgd = legend({'Before', 'After'}, 'Orientation', 'horizontal');
    lgd.Layout.Tile = 'south';
    title(tl, 'Channel-wise Performance');
    saveas(fig1, fullfile(results_dir, 'channel_wise_metrics.png'));

    %% Room-wise Metrics with single legend
    num_rooms = length(unique_rooms);
    fig2 = figure;
    tl = tiledlayout(1, num_rooms, 'TileSpacing', 'compact', 'Padding', 'compact');
    for r = 1:num_rooms
        nexttile;
        data_before = zeros(1, length(fields));
        data_after = zeros(1, length(fields));
        for i = 1:length(fields)
            data_before(i) = metrics_before.(unique_rooms{r}).(fields{i});
            data_after(i) = metrics_after.(unique_rooms{r}).(fields{i});
        end
        bar([data_before' data_after']);
        title(sprintf('Room: %s', unique_rooms{r}));
        ylabel('Score');
        ylim([0 1]);
        xticks(1:length(fields));
        xticklabels(fields);
        xtickangle(45);
        grid on;
    end
    lgd = legend({'Before', 'After'}, 'Orientation', 'horizontal');
    lgd.Layout.Tile = 'south';
    title(tl, 'Room-wise Performance');
    saveas(fig2, fullfile(results_dir, 'room_wise_metrics.png'));

    %% Average Metrics
    fig3 = figure;
    avg_before = zeros(1, length(fields));
    avg_after = zeros(1, length(fields));
    for i = 1:length(fields)
        avg_before(i) = metrics_before.Average.(fields{i});
        avg_after(i) = metrics_after.Average.(fields{i});
    end
    bar([avg_before' avg_after']);
    title('Average Performance Metrics');
    ylabel('Score');
    ylim([0 1]);
    xticks(1:length(fields));
    xticklabels(fields);
    xtickangle(45);
    legend('Before', 'After', 'Location', 'northoutside', 'Orientation', 'horizontal');
    grid on;
    saveas(fig3, fullfile(results_dir, 'average_metrics.png'));
end
