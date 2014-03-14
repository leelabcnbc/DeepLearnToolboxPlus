function patches = contrast_normalization(patches,epsilon)
% CONTRAST_NORMALIZATION normalize for contrast
%  
%   copy from kmeans_demo.m by Adam Coates
%   http://www.stanford.edu/~acoates/papers/kmeans_demo.tgz
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 14-Mar-2014 10:54:20 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : contrast_normalization.m 

assert(numel(size(patches)) == 2);

if nargin < 2
    epsilon = 10;
end

% normalize for contrast
patches = bsxfun(@rdivide, bsxfun(@minus, patches, mean(patches,2)), sqrt(var(patches,[],2)+epsilon));

end



% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [contrast_normalization.m] ======  
