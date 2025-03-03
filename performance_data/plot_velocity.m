clear; clc; close all;

% Read the CSV file
data = readtable('velocity_data.csv');

% Extract time and velocity data
time = data.time;
velocity = data.velocity;

% Create a figure with appropriate size
figure('Position', [100, 100, 900, 600]);

% Create a smooth curve using spline interpolation
time_fine = linspace(min(time), max(time), 500);
velocity_smooth = spline(time, velocity, time_fine);

% Define the terrain time ranges and other parameters
terrain_starts = [0, 5.45, 19.8, 29.65, 42.9, 49.45, 59.4, 70.4];
terrain_ends   = [5.45, 19.8, 29.65, 42.9, 49.45, 59.4, 70.4, 78.2];
terrain_names  = {'Flat', 'Ramp', 'Trench', 'Beam', 'Stairs', 'Slippery', 'Obstacles', 'Flat'};
terrain_colors = {[0.2 0.8 0.2], [1.0 1.0 0.2], [0.2 0.6 1.0], [1.0 0.5 0.0], ...
                  [0.8 0.4 0.8], [1.0 0.2 0.2], [0.5 0.5 0.5], [0.2 0.8 0.2]};

% Create patches for each terrain
hold on;
patch_handles = gobjects(length(terrain_starts), 1);
for i = 1:length(terrain_starts)
    start_time = terrain_starts(i);
    end_time = terrain_ends(i);
    
    x_patch = [start_time, end_time, end_time, start_time];
    y_patch = [min(velocity)-0.1, min(velocity)-0.1, max(velocity)+0.1, max(velocity)+0.1];
    
    patch_handles(i) = patch(x_patch, y_patch, terrain_colors{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(patch_handles(i), 'bottom');
end

% Plot the velocity line on top
plot(time_fine, velocity_smooth, 'LineWidth', 2, 'Color', [0,0,0]);

% Add grid and labels
xlabel('Time (s)', 'FontSize', 19);
ylabel('Velocity (m/s)', 'FontSize', 19);

% Customize the appearance
set(gca, 'FontSize', 19);

% Add vertical lines at terrain transitions
for i = 2:length(terrain_starts)
    transition_time = terrain_starts(i);
    
end

% Create legend entries for terrains
unique_names = unique(terrain_names, 'stable');
legend_handles = gobjects(length(unique_names), 1);
legend_names = cell(length(unique_names), 1);

% Create invisible square markers (boxes) for legend entries with matching transparency.
% Increase the marker size to 1000 for better visibility.
j = 1;
for i = 1:length(terrain_names)
    terrain_name = terrain_names{i};
    color = terrain_colors{i};
    
    % Check if this terrain was already added to the legend
    if i == 1 || ~any(strcmp(terrain_name, legend_names(1:j-1)))
        legend_handles(j) = patch([0, 0.5, 0.5, 0], [0, 0, 0.25, 0.25], color, ...
                              'FaceAlpha', 0.3, 'EdgeColor', 'none');
        legend_names{j} = terrain_name;
        j = j + 1;
    end
end

% Add horizontal legend for terrain types with non-empty entries
%legend_handle = legend(legend_handles(1:j-1), legend_names(1:j-1), ...
    %'Location', 'eastoutside', 'FontSize', 16, 'Orientation', 'horizontal');
%legend_handle.ItemTokenSize = [15, 15];

% Adjust axis limits
ylim([min(velocity)-0.05, max(velocity) + 0.05]);

set(findall(gcf, 'type', 'text'), 'FontName', 'Times New Roman', 'FontSize', 19);


% Create patches for each terrain and add text labels
hold on;
patch_handles = gobjects(length(terrain_starts), 1);
for i = 1:length(terrain_starts)
    start_time = terrain_starts(i);
    end_time = terrain_ends(i);
    
    % Define patch corners
    x_patch = [start_time, end_time, end_time, start_time];
    y_patch = [min(velocity)-0.1, min(velocity)-0.1, max(velocity)+0.1, max(velocity)+0.1];
    
    % Create the colored patch with transparency
    patch_handles(i) = patch(x_patch, y_patch, terrain_colors{i}, 'FaceAlpha', 0.2, 'EdgeColor', 'none');
    uistack(patch_handles(i), 'bottom');
    
    % Calculate the center of the patch (horizontally)
    x_label = (start_time + end_time) / 2;
    % Position the label near the top of the patch.
    % Adjust y_label as needed to avoid overlapping with the velocity plot.
    y_label = max(velocity) + 0.02;
    
    % Add the text label with desired properties
    text(x_label, y_label, terrain_names{i}, 'HorizontalAlignment', 'center', ...
         'FontName', 'Times New Roman', 'FontSize', 19, 'Color', 'k');
end
