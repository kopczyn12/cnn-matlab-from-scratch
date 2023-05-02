function y = softmax(x)
%Softmax function for distribuation to classify
exponent = exp(x);
y = exponent / sum(exponent);
end