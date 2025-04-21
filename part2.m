%%% 1. Initial cleanup, add paths and load in data
%%% DON'T CHANGE
clearvars;
close all;
addpath('../data');
addpath('../helper');
load lab3cardata.mat;

%%% 2. Declare total number of input neurons, hidden layer neurons and output
%%% neurons

%%% DON'T CHANGE
input_neurons = 6;

%%% This we can change
hidden_neurons = 2;

%%% DON'T CHANGE
output_neurons = 4;

%%% 3. Compute the total weights between the input and hidden layer and
%%% the hidden layer and output layer.  Also compute the total amount
%%% of weights
%%% DON'T CHANGE
total_weights_W1 = (input_neurons + 1)*hidden_neurons;
total_weights_W2 = (hidden_neurons + 1)*output_neurons;
total_weights = total_weights_W1 + total_weights_W2;

%%% 4. Create the initial parameter vector of weights
%%% DON'T CHANGE
rng(123);
e_init_1 = sqrt(6) / sqrt(input_neurons + hidden_neurons);
e_init_2 = sqrt(6) / sqrt(hidden_neurons + output_neurons);
initial_vec = zeros(total_weights,1);
initial_vec(1:total_weights_W1) = 2*e_init_1*rand(total_weights_W1,1) - e_init_1;
initial_vec(total_weights_W1 + 1:end) = 2*e_init_2*rand(total_weights_W2,1) - e_init_2;

%%% 5. Set total number of iterations
%%% DON'T CHANGE
N = 400;

%%% 6. Regularization parameter - This you can change
lambda = 0;

%%% 7. Declare optimization settings
%%% PLACE YOUR CODE HERE
options = optimset('GradObj', 'on', 'MaxIter', N);
%%% 8. Find optimal weights
%%% PLACE YOUR CODE HERE
%%% MAKE SURE THE OUTPUT WEIGHT PARAMETER VECTOR IS STORED IN A VARIABLE CALLED weights
costFunc = @(p) costFunction_NN_reg(Xtrain, Ytrain, lambda, ...
                    input_neurons, hidden_neurons, output_neurons, p);

weights = fmincg(costFunc, initial_vec, options);
%%% 9. Extract out the final weight matrices
%%% DON'T CHANGE
W1 = reshape(weights(1:total_weights_W1), hidden_neurons, input_neurons + 1).';
W2 = reshape(weights(total_weights_W1+1:end), output_neurons, hidden_neurons + 1).';
%%% 10. Compute predictions for training and testing data
%%% PLACE YOUR CODE HERE
Ytrain_pred = forward_propagation(Xtrain, W1, W2);  % Prediction scores on training data
train_classes = predict_class(Ytrain_pred);         % Final predicted class labels

Ytest_pred = forward_propagation(Xtest, W1, W2);    % Prediction scores on test data
test_classes = predict_class(Ytest_pred);           % Final predicted class labels
%%% 11. Compute classification accuracy for training and testing data
%%% PLACE YOUR CODE HERE
train_accuracy = sum(train_classes == Ytrain) / length(Ytrain) * 100;
test_accuracy = sum(test_classes == Ytest) / length(Ytest) * 100;


fprintf('Training Accuracy: %.2f%%\n', train_accuracy);
fprintf('Testing Accuracy: %.2f%%\n', test_accuracy);
