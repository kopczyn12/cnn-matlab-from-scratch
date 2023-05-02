clear all

Images = loadImg('t10k-images-idx3-ubyte');
Images = reshape(Images, 28, 28, []);
Labels = loadLabels('t10k-labels-idx1-ubyte');
Labels(Labels == 0) = 10;    % 0 --> 10

% Random seed - to keep weights initiliazed randomly the same value
rng(1);

% Learning 
%Initialization of weights,specialized for ReLU
W_conv = 1e-2*randn([9 9 20]);
W_pool = (2*rand(100, 2000) - 1) * sqrt(6) / sqrt(360 + 2000);
W_out = (2*rand( 10,  100) - 1) * sqrt(6) / sqrt( 10 +  100);

X = Images(:, :, 1:8000);
D = Labels(1:8000);

for epoch = 1:10
  answer = sprintf('%d epoch of 10', epoch);
  disp(answer)
  [W_conv, W_pool, W_out] = training(W_conv, W_pool, W_out, X, D);
end

% Saving model - to later get from that onnx format in python 
save('myConv.mat');
net = ('myConv.mat');


% Testing process
X = Images(:, :, 8001:10000);
D = Labels(8001:10000);

testing(W_conv, W_pool, W_out, X, D);


