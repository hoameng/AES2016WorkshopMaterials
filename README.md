# Welcome to the AES 2016 workshop on EEG Analysis and Seizure Detection!<br />
Please clone the repository or download the entire zip file before the workshop (clone/download -> download zip)<br />
Included files:<br />
Code/<br />
  ieegTrainFeats.m      - master script to extract features from data on ieeg.org, train model, test model, and upload predictions<br />
  feat_LineLength.m     - extract features based on line length<br />
  feat_freqcorr.m       - extract features based on winning Kaggle solution<br />
  IEEGTutorial.m        - tutorial for using ieeg.org MATLAB Toolbox<br />
  plotSeizureResults.m  - Generate rastor plot of detections for all layers<br />
  detect_seizures.m     - example script for detecting seizures on a separate dataset using saved model<br />
Tools/<br />
  ieeg-cli-1.13         - ieeg.org Command line toolbox to upload and interact with stored .MEF files<br />
  ieeg-matlab-1.13.2    - ieeg.org MATLAB toolbox<br />
  portal-matlab-tools   - partial port of https://github.com/ieeg-portal/portal-matlab-tools<br />
  
