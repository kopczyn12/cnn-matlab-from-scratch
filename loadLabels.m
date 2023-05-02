function labels = loadLabels(path)
%loading labels of MNIST dataset
fp = fopen(path, 'rb');
assert(fp ~= -1, ['Cannot load a file  ', path, '']);

magic = fread(fp, 1, 'int32', 0, 'ieee-be');
assert(magic == 2049, ['Wrong count ', path, '']);

numLabels = fread(fp, 1, 'int32', 0, 'ieee-be');
labels = fread(fp, inf, 'unsigned char');
assert(size(labels,1) == numLabels, 'Labels do not fit ');

fclose(fp);

end