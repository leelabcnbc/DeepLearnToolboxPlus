function images = contrast_normalization_3D(images, epsilon)
% CONTRAST_NORMALIZATION_3D normalize variance of each image separately.
%
%   a wrapper to pass the 2D patches to contrastNormalization.
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 04-Mar-2014 16:33:40 $
%% DEVELOPED : 8.1.0.604 (R2013a)
%% FILENAME  : contrast_normalization_3D.m

import preprocessing.*

if nargin < 2
    epsilon = 10;
end

numImages = size(images,3);
imageSize = [size(images,1), size(images,2)];
imagesReshaped = reshape(images,prod(imageSize),numImages);
imagesReshaped = imagesReshaped';
images = contrast_normalization(imagesReshaped,epsilon);
images = images';
images = reshape(images,[imageSize numImages]);

% code for debug
% images3 = zscore(imagesReshaped');
% 
% images3 = reshape(images3,[imageSize numImages]);
% diff_two_approach = abs(images3-images);
% diff_two_approach = diff_two_approach(:);
% if epsilon == 0
%     assert( min(diff_two_approach) < 1e-10);
%     assert( max(diff_two_approach) < 1e-10);
% end

end







% Created with NEWFCN.m by Frank González-Morphy
% Contact...: frank.gonzalez-morphy@mathworks.de
% ===== EOF ====== [contrast_normalization_3D.m] ======
