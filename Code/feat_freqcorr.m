
function feats = feat_freqcorr(data, fs)

    if ~any(any(isnan(data))) 
        %fft magnitudes from 1-47Hz
        freqRange = 1:47;
        P = pmtm(data,[],size(data,1),fs);
        feats = log10(abs(P(freqRange,:)))';
        feats = reshape(feats, 1,[]);
        
        %fft correlation + eigenvalues
        fc = corrcoef(zscore(P)); %normalize
        fc_eig = eig(fc);
        fc = triu(fc,1);
        fc = fc(fc~=0)';
        
        feats = [feats fc fc_eig'];

        %time correlation + eigenvalues
        tc = corrcoef(zscore(data)); %normalize
        tc_eig = eig(tc);
        tc = triu(tc,1);
        tc = tc(tc~=0)';
        
        feats = [feats tc tc_eig'];
    else
        feats = NaN(1,1024);
    end
    
end