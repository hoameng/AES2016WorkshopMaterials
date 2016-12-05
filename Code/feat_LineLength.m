function feat = feat_LineLength(x,fs)
%% Function will extract line length 
% Written by Hoameng Ung
% hoameng.ung@gmail.com
% University of Pennsylvania

chan_feat = sum(abs(diff(x, 1, 1)), 1);
feat = nanmean(chan_feat);

end
