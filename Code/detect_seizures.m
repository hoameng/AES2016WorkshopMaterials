
addpath(genpath('Z:\public\USERS\hoameng/Libraries/ieeg-matlab-1.13.2'));
%addpath('~/gdriveshort/Libraries/Utilities/hline_vline');
addpath(genpath('Z:\public\USERS\hoameng\Projects\p05-IEEGPortalToolbox/portalGit/Analysis'))
addpath(genpath('Z:\public\USERS\hoameng\Projects\p05-IEEGPortalToolbox/portalGit/Utilities'))
%javaaddpath('Z:\public\USERS\hoameng/Libraries/ieeg-matlab-1.13.2/IEEGToolbox/lib/ieeg-matlab.jar');

%params = initialize_task_humanNV;
params = initialize_task;

% Load data
session = loadData(params);
% 
% GENERATE TABLE
%find length of each dataset
subj = cell(numel(session.data),1);
dR = zeros(numel(session.data),1);
for i = 1:numel(session.data)
    subj{i} = session.data(i).snapName;
    dR(i) = session.data(i).rawChannels(1).get_tsdetails.getDuration/1e6/60/60/24;
end

channelIdxs = cell(numel(session.data),1);
for i = 1:numel(session.data)
    channelIdxs{i} = [1 3];
end

%% READ DATA AND RUN DETECTION
winLen = 2;
winDisp = 2;
load('rfmodel1000.mat')
for i = 2:numel(subj)
    x = getAllData(session.data(i),[1 3],12*3600);
    fs = session.data(i).sampleRate;

    %calculate features
    %out = [out cell2mat(runFuncOnWin(dat,fs,winLen,winDisp,@featFn))];
    out = cell2mat(runFuncOnWin(x,fs,winLen,winDisp,@calc_features));
    save(sprintf('%s_feats',session.data(i).snapName),'out');
end

[Ypred,Yscore]= predict(rfmodel,out);
Ypred = str2num(cell2mat(Ypred));
c = conv(Ypred,ones(1,10)*1/10,'same');
numel(find(c>.9))
