function opts = expandOpts(opts, optionLength)
% EXPANDOPTS expand everything in opts to vectors. 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 03-Mar-2014 15:27:26 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : expandOpts.m 

fields = fieldnames(opts);

for iField = 1:numel(fields)
    fieldThis = fields{iField};
    assert(isscalar(opts.(fieldThis)) || numel(opts.(fieldThis))==optionLength);
    
    if isscalar(opts.(fieldThis))
        opts.(fieldThis) = repmat(opts.(fieldThis),optionLength,1);
    end
end

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [expandOpts.m] ======  
