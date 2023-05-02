function [W_conv, W_pool, W_out] = testing(W_conv, W_pool, W_out, in, correct_out)
% Testing function to return accuracy of classifier.
acc = 0;
N   = length(correct_out);
disp('Every 200 iterations accuracy: ')
for k = 1:N
  x = in(:, :, k);                   % Input,           28x28

  y1 = convolution(x, W_conv);                 % Convolution,  20 9x9
  y2 = ReLU(y1);                    %
  y3 = pooling(y2);                    % Pool,         20 10x10
  y4 = reshape(y3, [], 1);          %                     
  v5 = W_pool*y4;                       % ReLU,              
  y5 = ReLU(v5);                    %
  v  = W_out*y5;                       % Softmax,            
  y  = softmax(v);                  %

  [~, i] = max(y);
  if i == correct_out(k)
    acc = acc + 1;
    if (mod(k,200)) == 0
       temp = acc/N;
       disp(temp)
    end
  end
end

acc = acc / N;
fprintf('Final Accuracy is %f\n', acc*100);

end