clc; clear; close all;

% Read CSV data
data = readtable('robustness_data.csv');

% Extract data columns
sensorNoise = data.sensor_noise;      % in radians
surfaceNoise = data.surface_noise;    % in meters
avgSteps = data.avg_steps;            % average steps per episode
stdSteps = data.std_steps;            % standard deviation per episode

% Compute percentage per episode.
% Maximum per episode is 72, so percentage = (avgSteps/72)*100.
percentage = (avgSteps / 72) * 100;
percentageError = (stdSteps / 72) * 100;  % error in percentage

% Convert surface noise from meters to centimeters for x-axis
surfaceNoise_cm = surfaceNoise * 100;

% Get unique sensor noise levels
uniqueSensorNoise = unique(sensorNoise);

figure;
hold on;

% Loop through each sensor noise level
for i = 1:length(uniqueSensorNoise)
    % Find indices for the current sensor noise level
    idx = sensorNoise == uniqueSensorNoise(i);
    
    % Sort by surface noise values
    [x_sorted, sortIdx] = sort(surfaceNoise_cm(idx));
    y_sorted = percentage(idx);
    y_sorted = y_sorted(sortIdx);
    err_sorted = percentageError(idx);
    err_sorted = err_sorted(sortIdx);
    
    % Plot errorbars with markers
    errorbar(x_sorted, y_sorted, err_sorted, '-o', 'LineWidth', 2, 'MarkerSize', 5);
end

% Create legend strings converting sensor noise from radians to degrees
legendStrings = cell(length(uniqueSensorNoise), 1);
for i = 1:length(uniqueSensorNoise)
    degVal = uniqueSensorNoise(i) * 180 / pi;
    legendStrings{i} = sprintf('Sensor noise = %0.0f°', degVal);
end

% Set labels, title, legend, and tick properties
xlabel('Surface noise (cm)', 'FontSize', 19);
ylabel('Completed Steps (%)', 'FontSize', 19);
legend(legendStrings, 'FontSize', 19, 'FontName', 'Times New Roman', 'Location', 'best');
set(gca, 'FontSize', 19);
grid on;
ylim([30 110]);
set(findall(gcf, 'type', 'text'), 'FontName', 'Times New Roman', 'FontSize', 19);
