%% Script to clip interictal, preictal, ictal segments from Dog Dataset
% from IEEG Portal 
% Note: This is a demonstration of real-time seizure detection, operating on one window at a time.
% Computational efficiency can be improved by batching feature extraction
% and classification

clear; clc;
addpath(genpath('../'))

% Once you've signed up and have been approved on the ieeg.org website, you can establish a key that will give you access permissions to connect to our server and interact with datasets'
% This creates a .bin file that contains your login (unencrypted)
% IEEGSession.createPwdFile('username','password')

%% ENTER OWN USERNAME AND PASSWORD FILE HERE
ieegUser = 'hoameng';
ieegPwd = 'hoa_ieeglogin.bin'; %created bin file

%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%

layerAppend = '';

%% FEATURE OPTIONS: 
% 1: 'linelength'
% 2: 'freqcorr'

featureOption = 2;

%% CLASSIFIER OPTIONS
% 1: Logistic Regression
% 2: Random Forest

classifierOption = 2;

%% Portal Data Set Options
%
% Empty space to configure IEEG Account Login, IEEG Snapshot, Time Conversion Constants
%
S2US = 1e6;
US2S = 1e-6;

snapshotName = 'I004_A0003_D001';
snapTrainPrefix = '-TrainingAnnots';
trainAnnLayerName = 'Train';
snapTestPrefix = '-TestingAnnots';
testAnnLayerName = 'Test';

%% Assign Feature Function
switch featureOption
    case 1
        featFn = @feat_LineLength;
        layerAppend = [layerAppend '-LL'];
    case 2
        featFn = @feat_freqcorr;
        layerAppend = [layerAppend '-FC'];
end

%% Grab training snapshot, extract features, train model
%
% Insert code to instantiate an IEEG Session pointing training data snapshot
%
trainSnapName = strcat(snapshotName, snapTrainPrefix);
session = IEEGSession(trainSnapName, ieegUser, ieegPwd);
dataset = session.data;

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%
fs = dataset.sampleRate;
dsDurSn = dataset.rawChannels(1).get_tsdetails.getDuration * US2S * fs;
dsDurSn = floor(dsDurSn);
channelsIdx = 1:numel(dataset.rawChannels);

%
% Select the "Train" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
[allAnn, allAnnUsec, allAnnChans] = getAnnotations(dataset, trainAnnLayerName);

%
% For Machine Learning, initialize matrices
% Matrix for features per clip, and training label per clip (interictal/ictal)
%
featMatr = cell(numel(allAnn), 1);
trainLabel = zeros(numel(allAnn), 1);

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute and store features for that clip
%    3. Store the binary label for that clip
%        (Clip label is stored as an annotation type for a given annotation layer)
%        (We use type 'NSZ' for interictal and 'SZ' for ictal)
%
for i = 1:numel(allAnn)
   fprintf('Features from clip %d of %d\n', i, numel(allAnn))

   % Get values for each
   snRange = allAnn(i).start * US2S * fs : allAnn(i).stop * US2S * fs;
   annData = getExtendedData(dataset, snRange, channelsIdx);

   % Compute features and add to feature matrix
   feat = featFn(annData,fs);
   featMatr{i} = feat;

   % Save label for each
   if strcmp(allAnn(i).type, 'NSZ')
       trainLabel(i, 1) = 1;
   end
   if strcmp(allAnn(i).type, 'SZ')
       trainLabel(i, 1) = 2;
   end
end

featMatr = cell2mat(featMatr);
%remove any examples with NaN features
[r c] = find(isnan(featMatr));
r = unique(r);
featMatr(r,:) = [];
trainLabel(r) = [];

%
% With clip interictal/ictal labels and feature set, train ML algorithm
%
switch classifierOption
    case 1
        % Logistic Regression
        fprintf('\nTraining Logistic Regression\n')
        model = mnrfit(featMatr, trainLabel);
        layerAppend = [layerAppend '-LR'];

    case 2
        % Random Forest
        fprintf('\nTraining Random Forest\n')
        model = TreeBagger(3000,featMatr, trainLabel,'Method','Classification','OOBPrediction','on','OOBPredictorImportance','on');
        layerAppend = [layerAppend '-RF'];

        [yhat,scores] = oobPredict(model);
        [conf, classorder] = confusionmat(categorical(trainLabel), categorical(yhat));
end


%% Grab testing snapshot, extract features, predict using model, upload prediction
%
% Insert code to instantiate an IEEG Session pointing testing data snapshot
% You may append the snapshot to the current session, rather than restarting the session
%
testSnapName = strcat(snapshotName, snapTestPrefix);
session.openDataSet(testSnapName);

%
% Insert code to Retrieve dataset sampling frequency, signal length, channel indices
%
dataset = session.data(2);
Fs = dataset.sampleRate;
dsDurSn = dataset.rawChannels(1).get_tsdetails.getDuration * US2S * fs;
dsDurSn = floor(dsDurSn);
channelsIdx = 1:numel(dataset.rawChannels);

%
% Select the "Test" Annotation layer, and grab all associated annotations
% Retrieve and store annotation object for every annotation in the layer
%
[allAnn, allAnnUsec, allAnnChans] = getAnnotations(dataset, testAnnLayerName);

%
% For Machine Learning, initialize matrices
% Matrix for predicted label per clip
%
predLabel = zeros(numel(allAnn), 1);

%
% Iterate over each annotation object
% For each annotation:
%    1. Retrieve time range and associated signal clip
%    2. Compute features for that clip
%    3. Use ML algorithm to predict clip label (interictal/ictal)
%    4. Store the binary label for that clip
%
for i = 1:numel(allAnn)
   fprintf('Predicting clip %d of %d', i, numel(allAnn))

   % Get values for each
   snRange = allAnn(i).start * US2S * fs : allAnn(i).stop * US2S * fs;
   annData = getExtendedData(dataset, snRange, channelsIdx);

   % Compute features
   feat = featFn(annData,fs);

   switch classifierOption
       case 1
           % Predict logistic regression
           phat = mnrval(B, feat);
           [~, predLabel(i, 1)] = max(phat);
   
       case 3
           phat = predict(model,feat);
           predLabel(i,1) = str2num(phat{1});
   end
   
   if(predLabel(i,1) == 2)
       fprintf('.... Detected seizure !!!\n')
   else
       fprintf('\n')
   end
end

%
% Retrieve the predicted labels that indicate seizures
% Grab the start and stop time for each of the predicted seizure labels 
%   (corresponding to original annotation start/stop time)
% Also, retrieve the channels on which the prediction occurred (in our case all channels)
% Use the uploadAnnotations tool to
%    1. Create an annotation layer with the same name as your ieegUser name
%    2. Post seizure annotations with the annotation type: 'seizure'
%
seizures = allAnnUsec(predLabel(:,1) == 2, :);
channels = allAnnChans(predLabel(:,1)== 2)';
layerName = [ieegUser layerAppend];
uploadAnnotations(dataset,layerName,seizures,channels,layerName,'overwrite');

