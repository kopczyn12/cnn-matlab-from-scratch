function [W_conv, W_pool, W_out] = training(W_conv, W_pool, W_out, in, correct_out)
% function returns weights (training is done by backprop method)
%where W_conv, W_pool, W_out (are the layers which contain weights in this architecture)
%are the conv filters, pooling hidden-layer weight matrix,
%W_pool and Wo contain the connection weights of the classification layer,
%W_conv
%about convolution layer, which are used for convolution filters for image
%processing
%and hidden output layer weight matrix respectively.
%in and correct_out are input and correct output from the trainging data.
% alpha and beta  - hyperparameters for momentum method (if beta big ->
% alpha small like here - rule of momemntum method)
alpha = 0.01;
beta  = 0.95;

%Initialize the vecs for collect weights and update them in the way of
%momentum method
momentum_pool = zeros(size(W_pool));
momentum_out = zeros(size(W_out));
momentum_conv = zeros(size(W_conv));


N = length(correct_out);

batch_size = 100;  % number of batches
batch_list = 1:batch_size:(N-batch_size+1); % the location of the first training data point to be brought into the minibatch

% One epoch loop
%
for batch = 1:length(batch_list)
  dW_conv = zeros(size(W_conv));
  dW_pool = zeros(size(W_pool));
  dW_out = zeros(size(W_out));
  
  % Mini-batch loop
  %
  begin = batch_list(batch);
  for k = begin:begin+batch_size-1
    % Forward pass = inference
    %
    x  = in(:, :, k);               % Input,           28x28 = 784 input nodes
    y1 = convolution(x, W_conv);              % Convolution 20 9x9 filters
    y2 = ReLU(y1);                 %
    y3 = pooling(y2);                 % Pooling,   2x2 mean pooling process  
    y4 = reshape(y3, [], 1);       %
    v5 = W_pool*y4;                    % ReLU,             
    y5 = ReLU(v5);                 % Hidden
    v  = W_out*y5;                    % Softmax         
    y  = softmax(v);               %

    % One-hot encoding
    % Correct output should be stored in 10x1 vector in order to calculate
    % error.
    d = zeros(10, 1);
    d(sub2ind(size(d), correct_out(k), 1)) = 1;

    % Backpropagation the whole layers
    % Predicted - actual
    e      = d - y;                   % Output layer  
    delta  = e;                       % Error (Predicted - actual, cross-entropy)

    e5     = W_out' * delta;             % Hidden(ReLU) layer
    delta5 = (y5 > 0) .* e5;

    e4     = W_pool' * delta5;            % Pooling layer
    
    e3     = reshape(e4, size(y3));

    e2 = zeros(size(y2));           
    W3 = ones(size(y2)) / (2*2);
    %backpropagation in CNN is also convolution (reverse convolution)!
    for c = 1:20
      e2(:, :, c) = kron(e3(:, :, c), ones([2 2])) .* W3(:, :, c);
    end
    
    delta2 = (y2 > 0) .* e2;          % ReLU layer
  
    delta1_x = zeros(size(W_conv));       % Convolutional layer
    for c = 1:20
      delta1_x(:, :, c) = conv2(x(:, :), rot90(delta2(:, :, c), 2), 'valid');
    end
    
    dW_conv = dW_conv + delta1_x; 
    dW_pool = dW_pool + delta5*y4';    
    dW_out = dW_out + delta *y5';
  end 
  
  % Update weights - momentum method
  dW_conv = dW_conv / batch_size;
  dW_pool = dW_pool / batch_size;
  dW_out = dW_out / batch_size;
  
  momentum_conv = alpha*dW_conv + beta*momentum_conv;
  W_conv        = W_conv + momentum_conv;
  
  momentum_pool = alpha*dW_pool + beta*momentum_pool;
  W_pool        = W_pool + momentum_pool;
   
  momentum_out = alpha*dW_out + beta*momentum_out;
  W_out        = W_out + momentum_out;  
end

end