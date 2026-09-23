%% Plot MNI coordinates color-coded by patient/subject
% Expected spreadsheet columns:
% Subject, ID, ch, ch_orig, MNI_X, MNI_Y, MNI_Z

clearvars;
close all;
clc;

%% Paths
addpath(genpath('/Users/Timon/Documents/MATLAB/leaddbs'));
addpath(genpath('/Users/Timon/Documents/MATLAB/spm12'));

xlsx_file = 'coords_comb.xlsx';

if ~isfile(xlsx_file)
    % Also check the plot_coordinates subfolder.
    alternative_file = fullfile('plot_coordinates', 'coords_comb.xlsx');
    if isfile(alternative_file)
        xlsx_file = alternative_file;
    else
        error('Could not find coords_comb.xlsx in the current folder or plot_coordinates/.');
    end
end

%% Open Lead-DBS MNI figure
ea_mnifigure;

resultfig = gcf;
ax = gca;
hold(ax, 'on');

%% Load spreadsheet
opts = detectImportOptions(xlsx_file, 'VariableNamingRule', 'preserve');
data = readtable(xlsx_file, opts);

required_columns = {'Subject', 'ID', 'ch', 'ch_orig', ...
    'MNI_X', 'MNI_Y', 'MNI_Z'};

missing_columns = required_columns(~ismember(required_columns, data.Properties.VariableNames));
if ~isempty(missing_columns)
    error('Missing required column(s): %s', strjoin(missing_columns, ', '));
end

% Convert subject labels to strings so numeric and text identifiers both work.
subject_labels = string(data.('Subject'));

% Use ID as a fallback when Subject is empty.
id_labels = string(data.('ID'));
empty_subject = ismissing(subject_labels) | strlength(strtrim(subject_labels)) == 0;
subject_labels(empty_subject) = id_labels(empty_subject);

% Convert coordinates to numeric in case Excel imported them as strings.
mni_x = makeNumeric(data.('MNI_X'));
mni_y = makeNumeric(data.('MNI_Y'));
mni_z = makeNumeric(data.('MNI_Z'));

% Remove rows with missing subject names or invalid coordinates.
valid_rows = ~(ismissing(subject_labels) | ...
    strlength(strtrim(subject_labels)) == 0 | ...
    isnan(mni_x) | isnan(mni_y) | isnan(mni_z));

subject_labels = subject_labels(valid_rows);
mni_x = mni_x(valid_rows);
mni_y = mni_y(valid_rows);
mni_z = mni_z(valid_rows);

if isempty(subject_labels)
    error('No valid subject-coordinate rows were found.');
end

%% Subject colors
% 'stable' preserves the order in which subjects occur in the spreadsheet.
subjects = unique(subject_labels, 'stable');
n_subjects = numel(subjects);

% Distinct patient colors.
colors = lines(n_subjects);

%% Sphere settings
sphere_radius = 1;       % Radius in MNI millimetres
sphere_resolution = 12; % Higher values produce smoother spheres
face_alpha = 0.90;

[XS, YS, ZS] = sphere(sphere_resolution);

% Optional coordinate filter. Set to false to plot every coordinate.
apply_y_filter = true;
maximum_y = 20;

legend_handles = gobjects(n_subjects, 1);
legend_labels = cell(n_subjects, 1);

n_plotted = 0;

%% Plot each patient
for s = 1:n_subjects
    current_subject = subjects(s);
    current_color = colors(s, :);

    subject_mask = subject_labels == current_subject;
    subject_coords = [mni_x(subject_mask), ...
        mni_y(subject_mask), ...
        mni_z(subject_mask)];

    if apply_y_filter
        subject_coords = subject_coords(subject_coords(:, 2) <= maximum_y, :);
    end

    if isempty(subject_coords)
        continue;
    end

    % Invisible dummy marker provides exactly one clean legend entry.
    legend_handles(s) = plot3(ax, NaN, NaN, NaN, 'o', ...
        'MarkerSize', 8, ...
        'MarkerFaceColor', current_color, ...
        'MarkerEdgeColor', 'k', ...
        'LineStyle', 'none');

    legend_labels{s} = char(current_subject);

    for k = 1:size(subject_coords, 1)
        x0 = subject_coords(k, 1);
        y0 = subject_coords(k, 2);
        z0 = subject_coords(k, 3);

        surf(ax, ...
            sphere_radius * XS + x0, ...
            sphere_radius * YS + y0, ...
            sphere_radius * ZS + z0, ...
            'FaceColor', current_color, ...
            'EdgeColor', 'none', ...
            'FaceAlpha', face_alpha, ...
            'HandleVisibility', 'off');

        n_plotted = n_plotted + 1;
    end
end

%% Remove empty legend entries
valid_legend = isgraphics(legend_handles);
legend_handles = legend_handles(valid_legend);
legend_labels = legend_labels(valid_legend);

if ~isempty(legend_handles)
    legend(ax, legend_handles, legend_labels, ...
        'Location', 'bestoutside', ...
        'FontSize', 9, ...
        'Interpreter', 'none');
end

%% Figure formatting
xlabel(ax, 'MNI X (mm)');
ylabel(ax, 'MNI Y (mm)');
zlabel(ax, 'MNI Z (mm)');
title(ax, 'MNI Coordinates Color-Coded by Patient', 'Interpreter', 'none');

axis(ax, 'equal');
grid(ax, 'on');
view(ax, 3);
% lighting(ax, 'gouraud');
% camlight(ax, 'headlight');

hold(ax, 'on');

fprintf('Plotted %d coordinates from %d subjects.\n', ...
    n_plotted, numel(legend_labels));

%% Local helper function
function values = makeNumeric(input_values)
% Convert numeric, cell, categorical, or string table variables to doubles.
if isnumeric(input_values)
    values = double(input_values);
else
    values = str2double(string(input_values));
end
end
