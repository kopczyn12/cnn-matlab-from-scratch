function imgs = loadImg(path)
%loading images from MNIST dataset.
fp = fopen(path, 'rb');
assert(fp ~= -1, ['Cannot load a file ', path, '']);
magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2051, ['Wrong load', path, '']);
numOfImgs = fread(fp, 1, 'int32', 0, 'ieee-be');
numOfCols = fread(fp, 1, 'int32', 0, 'ieee-be');
numOfRows = fread(fp, 1, 'int32', 0, 'ieee-be');

imgs = fread(fp, inf, 'unsigned char=>unsigned char');
imgs = reshape(imgs, numOfCols, numOfRows, numOfImgs);
imgs = permute(imgs,[2 1 3]);
fclose(fp);

imgs = reshape(imgs, size(imgs, 1) * size(imgs, 2), size(imgs, 3));
imgs = double(imgs) / 255;

end

