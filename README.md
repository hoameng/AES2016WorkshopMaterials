# Welcome to the AES 2016 workshop on EEG Analysis and Seizure Detection!<br />
Please clone the repository or download the entire zip file before the workshop (clone/download -> download zip)<br />

Sign up for an account at http://main.ieeg.org. Put "AES2016" in the "Data Use" field. For institutional contact, is it OK to put your own contact information. <br />
Included files:<br />

Code/<br />
  *ieegTrainFeats.m*      - master script to extract features from data on ieeg.org, train model, test model, and upload predictions<br />
  *feat_LineLength.m*     - extract features based on line length<br />
  *feat_freqcorr.m*       - extract features based on winning Kaggle solution<br />
  *IEEGTutorial.m*        - tutorial for using ieeg.org MATLAB Toolbox<br />
  *plotSeizureResults.m*  - Generate rastor plot of detections for all layers<br />
  
Tools/<br />
  *ieeg-cli-1.13*         - ieeg.org Command line toolbox to upload and interact with stored .MEF files<br />
  *ieeg-matlab-1.13.2*    - ieeg.org MATLAB toolbox<br />
  *portal-matlab-tools*   - partial port of https://github.com/ieeg-portal/portal-matlab-tools<br />
  
Please contact hoameng@mail.med.upenn.edu with any questions or comments.
