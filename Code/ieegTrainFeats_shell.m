%%% Script to train/test machine learning algorithms to detect interictal/ictal segments

clear; clc;
addpath(genpath('../'))

%%% Portal Data Set Options
%
% Empty space to configure IEEG Account Login, IEEG Snapshot, Time Conversion Constants
%


%%% Grab training snapshot, extract features, train model
%
% Insert code to instantiate an IEEG Session pointing training data snapshot
%

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%

%
% Select the "Train" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
%

%
% For Machine Learning, initialize matrices
% Matrix for features per clip, and training label per clip (interictal/ictal)
%

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute and store features for that clip
%    3. Store the binary label for that clip
%        (Clip label is stored as an annotation type for a given annotation layer)
%        (We use type 'NSZ' for interictal and 'SZ' for ictal)
%

%
% With clip interictal/ictal labels and feature set, train ML algorithm
%

%%% Grab testing snapshot, extract features, predict using model, upload prediction
%
% Insert code to instantiate an IEEG Session pointing testing data snapshot
% You may append the snapshot to the current session, rather than restarting the session
%

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%

%
% Select the "Test" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
%

%
% For Machine Learning, initialize matrices
% Matrix for predicted label per clip
%

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute features for that clip
%    3. Use ML algorithm to predict clip label (interictal/ictal)
%    4. Store the binary label for that clip
%

%
% Retrieve the predicted labels that indicate seizures
% Grab the start and stop time for each of the predicted seizure labels 
%   (corresponding to original annotation start/stop time)
% Also, retrieve the channels on which the prediction occurred (in our case all channels)
% Use the uploadAnnotations tool to
%    1. Create an annotation layer with the same name as your ieegUser name
%    2. Post seizure annotations with the annotation type: 'seizure'
%
