function y = pooling(x)
%function returns the image after 2x2 mean pooling process.
[xrow, xcol, numOfFilters] = size(x);
y = zeros(xrow/2, xcol/2, numOfFilters);

for k = 1:numOfFilters
    filter = ones(2) / (2*2);
    image = conv2(x(:, :, k), filter, 'valid');
    y(:, :, k) = image(1:2:end, 1:2:end);
end
end