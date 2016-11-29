function feat = feat_LineLength(x,fs)

chan_feat = sum(abs(diff(x, 1, 1)), 1);
feat = mean(chan_feat);

end
