function [X, unwhitenMatrix, M, D] = whiten_ZCA(X, epsilon)
% WHITEN_ZCA
%
%   copy from kmeans_demo.m by Adam Coates
%   http://www.stanford.edu/~acoates/papers/kmeans_demo.tgz   
%
% X is a NxP matrix, N is number of observations, P is number of features
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 14-Mar-2014 11:10:26 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : whiten_ZCA.m

assert(numel(size(X)) == 2);

if nargin < 2
    epsilon = 0.1;
end

C = cov(X);
M = mean(X);
[V,D,~] = svd(C);
P = V * diag(sqrt(1./(diag(D) + epsilon))) * V';
X = bsxfun(@minus, X, M) * P;

unwhitenMatrix = V * diag( 1./(  sqrt(1./(diag(D) + epsilon)) )  ) * V';


end








% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [whiten_ZCA.m] ======
