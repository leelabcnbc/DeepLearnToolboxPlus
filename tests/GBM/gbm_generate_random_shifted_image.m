function [xData, yData] = gbm_generate_random_shifted_image(N,sideSize,probOn,debug)
% GBM_GENERATE_RANDOM_SHIFTED_IMAGE ... 
%  
%   ... 
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 16-May-2014 17:31:55 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : gbm_generate_random_shifted_image.m 

if nargin < 2 || isempty(sideSize)
    sideSize = 10;
end

if nargin < 3 || isempty(probOn)
    probOn = 0.1;
end

if nargin < 4 || isempty(debug)
    debug = 0.1;
end

xData = zeros(N,sideSize*sideSize);
yData = zeros(N,sideSize*sideSize);

for iImage = 1:N
    imageOriginal = double(rand(sideSize+2,sideSize+2) < probOn);
    xDataInShape = imageOriginal(2:end-1,2:end-1);
    xData(iImage,:) = xDataInShape(:);
    
    horizontalShift = randi([-1, 1]);
    verticalShift = randi([-1, 1]);
    yDataInShape = imageOriginal(2+verticalShift:end-1+verticalShift,2+horizontalShift:end-1+horizontalShift);
    yData(iImage,:) = yDataInShape(:);
%     disp(iImage);
    
    if debug
        subplot(1,2,1);
        imagesc(xDataInShape); colormap gray;
        subplot(1,2,2);
        imagesc(yDataInShape); colormap gray;
        titlestr = sprintf('vertical: %d, horizontal: %d\n', verticalShift, horizontalShift);
        title(titlestr);
        pause;
    end
        
    
end
    
end








% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [gbm_generate_random_shifted_image.m] ======  
