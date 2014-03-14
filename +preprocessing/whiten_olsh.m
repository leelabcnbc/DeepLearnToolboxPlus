function [IMAGES] = whiten_olsh(images_in, avg_var)
% Whitening images (Olsh & Field)

% modified to customize average variance (although it's better to always
% zscore the data set after this procedure, IMO).
% Yimeng Zhang
% Computer Science Department, Carnegie Mellon University
% zym1010@gmail.com

if nargin < 2
    avg_var = 0.1;
end

image_size = min(size(images_in,1),size(images_in,2));
num_images = size(images_in,3);

if (size(images_in,1)~=size(images_in,2)) % better be square
    warning('width and height are not equal');
end

N = image_size;
M = num_images;

images_in = images_in(1:N,1:N,:); %this can do cropping.

[fx, fy] = meshgrid(-N/2:N/2-1,-N/2:N/2-1);
rho = sqrt(fx.*fx+fy.*fy);
f_0 = 0.4*N;
filt = rho.*exp(-(rho/f_0).^4);

IMAGES = zeros(N^2,M);

for i = 1:M
    image = images_in(:,:,i);  % you will need to provide get_image
    If = fft2(image);
    imagew = real(ifft2(If.*fftshift(filt)));
    IMAGES(:,i) = reshape(imagew,N^2,1);
end

IMAGES = sqrt(avg_var)*IMAGES/sqrt(mean(var(IMAGES)));

assert(abs(avg_var)-mean(var(IMAGES)) < 1e-6);

IMAGES = reshape(IMAGES,[N N M]);

% save MY_IMAGES IMAGES