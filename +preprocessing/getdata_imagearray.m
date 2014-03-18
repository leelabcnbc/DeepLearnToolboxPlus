function X = getdata_imagearray(IMAGES, winsize, num_patches, BUFF)
% GETDATA_IMAGEARRAY ... 
%  
%   copied & modified from Honglak Lee's sparse coding implementation
%   http://web.eecs.umich.edu/~honglak/softwares/nips06-sparsecoding.htm
%   http://web.eecs.umich.edu/~honglak/softwares/fast_sc.tgz
%
%   now, I return NxP array, where N is num_patches, P is winsize^2.
%
%
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

%% DATE      : 14-Mar-2014 21:22:52 $ 
%% DEVELOPED : 8.1.0.604 (R2013a) 
%% FILENAME  : getdata_imagearray.m 

num_images=size(IMAGES,3);
image_size1=size(IMAGES,1);
image_size2=size(IMAGES,2);
sz= winsize;

if (size(IMAGES,1)~=size(IMAGES,2))
    warning('patches better be square!');
end

if nargin < 4
    BUFF=4;
end

totalsamples = 0;
% extract subimages at random from this image to make data vector X
% Step through the images
X= zeros(sz^2, num_patches);
for i=1:num_images,

    % Display progress
    fprintf('[%d/%d]\n',i,num_images);

    this_image=IMAGES(:,:,i);

    % Determine how many patches to take
    getsample = floor(num_patches/num_images);
    if i==num_images, getsample = num_patches-totalsamples; end

    % Extract patches at random from this image to make data vector X
    for j=1:getsample
        r=BUFF+ceil((image_size1-sz-2*BUFF)*rand);
        c=BUFF+ceil((image_size2-sz-2*BUFF)*rand);
        totalsamples = totalsamples + 1;
        % X(:,totalsamples)=reshape(this_image(r:r+sz-1,c:c+sz-1),sz^2,1);
        temp =reshape(this_image(r:r+sz-1,c:c+sz-1),sz^2,1);
%         X(:,totalsamples) = temp - mean(temp);
        X(:,totalsamples) = temp; %keep things as original.
    end
end  

X = X';

end




% Created with NEWFCN.m by Frank González-Morphy 
% Contact...: frank.gonzalez-morphy@mathworks.de  
% ===== EOF ====== [getdata_imagearray.m] ======  
