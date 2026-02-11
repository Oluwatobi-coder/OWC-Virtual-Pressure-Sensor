%% Hydrodynamic Characterization and Virtual Pressure Sensing of an Oscillating Water Column (OWC)
% Purpose: This script processes experimental tank-test data from the Lir National Ocean 
%          Test Facility for a Marinet 2 Fixed OWC model. It processes raw 
%          time-series data, converting  pressure measurements to Pascals and 
%          applying moving-average filtering for noise reduction. The final cleaned datasets 
%          are segmented into training and validation sets for virtual pressure sensor modeling.
% Author:  Bello Oluwatobi
% Date:    September 22, 2025

clear; clc;

% setting the relative paths for handling the data files and 
% storing the processed files
scriptFolder = fileparts(mfilename('fullpath')); 
basePathDataFiles = fullfile(scriptFolder, 'Data_Files'); 
basePathProcessed = fullfile(scriptFolder, 'Processed_Data_Files');

% defining the indices for storing the training, 
% validation, and test datasets
trainIndices = [37:48, 50:51];   % Operational Sea States (Hs between 0.025m and 0.125m)
highIrrIndex  = 49;               % Highly Irregular Sea State (Hs = 0.158m)
normIndex = 5;                % Regular/Normal Wave (Hs = 0.025m)

fs = 100;                        % sampling frequency of the data (100 Hz)
trimSamples = 15 * fs;           % trimming transient startup noise (15 seconds at 100 Hz)


% iteratively loading and cleaning the datasets using the helper function.
trainTable = table(); % initializing table to store training data

% building the training dataset i.e. combination of multiple sea states
for i = 1:length(trainIndices)
    processed = loadAndClean(trainIndices(i), basePathDataFiles, trimSamples);
    if ~isempty(processed), trainTable = [trainTable; processed]; end
end

% building validation/test sets i.e. the irregular extreme sea state and the regular/normal wave state 
highIrregularTable = loadAndClean(highIrrIndex, basePathDataFiles, trimSamples);
normalTable = loadAndClean(normIndex, basePathDataFiles, trimSamples);

%% 3. Export to CSV (Formatted for Python/Machine Learning)
% Saving to the dedicated Processed Data folder
writetable(trainTable, fullfile(basePathProcessed, 'owc_training_data.csv'));
writetable(highIrregularTable, fullfile(basePathProcessed, 'owc_high_irregular_test_data.csv'));
writetable(normalTable, fullfile(basePathProcessed, 'owc_normal_test_data.csv'));

fprintf("Data preprocessing complete. Processed files saved to '%s'.\n", basePathProcessed);

% writing a helper function to handle raw data loading, scaling, and signal conditioning
function cleanTbl = loadAndClean(idx, basePathDataFiles, trimSamples)
    fileName = sprintf('Testdatas%02d.mat', idx); 
    fullPath = fullfile(basePathDataFiles, fileName);
    cleanTbl = [];
    
    if isfile(fullPath)
        S = load(fullPath);
        fields = fieldnames(S);
        DataStruct = S.(fields{1}); 
        raw_data = DataStruct.data;
        raw_time = DataStruct.time;
        
        % PHYSICAL CONVERSION: 
        % Combining redundant Pressure Sensors (Pr1/Pr2) and converting to Pa.
        P_avg = (raw_data(:,7)*100 + raw_data(:,8)*100) / 2; 
        
        % SIGNAL CONDITIONING:
        % Applying a 10-sample moving mean to smooth sensor noise while 
        % preserving hydrodynamic features.
        T = table(raw_time, movmean(raw_data(:,1),10), movmean(raw_data(:,4),10), ...
            movmean(P_avg,10), 'VariableNames', {'Time', 'WG1', 'WG6', 'P_Chamber'});
        
        T.TestID = repmat(idx, height(T), 1);
        
        % TRIMMING: Removing startup/shutdown transients (15s each)
        if height(T) > 2*trimSamples
            cleanTbl = T(trimSamples+1 : end-trimSamples, :);
        end
    end
end