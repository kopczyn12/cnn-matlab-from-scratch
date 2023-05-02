function y = convolution(x, W)
%function takes the input images, do the convolution and return the
%feature maps.
[wrow, wcol, numOfFilters] = size(W);
[xrow, xcol, ~         ] = size(x);

yrow = xrow - wrow + 1;
ycol = xcol - wcol + 1;

y = zeros(yrow, ycol, numOfFilters);

for k = 1:numOfFilters
    filter = W(:, :, k);
    filter = rot90(squeeze(filter), 2);
    y(:, :, k) = conv2(x, filter, 'valid');
end
end