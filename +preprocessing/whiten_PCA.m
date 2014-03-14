function [X, unwhitenMatrix, M, D] = whiten_PCA(X, epsilon)
% WHITEN_PCA ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 14-Mar-2014 11:10:35 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : whiten_PCA.m 

assert(numel(size(X)) == 2);

if nargin < 2
    epsilon = 0.1;
end

C = cov(X);
M = mean(X);
[V,D,~] = svd(C);

P = V * diag(sqrt(1./(diag(D) + epsilon)));
X = bsxfun(@minus, X, M) * P;

unwhitenMatrix = diag( 1./(  sqrt(1./(diag(D) + epsilon)) )  ) * V';

end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [whiten_PCA.m] ======  
