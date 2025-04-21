%%% 1. Clear all variables and close all figures
%%% DON'T CHANGE
clearvars;
close all;
addpath('../helper');

%%% 2. Input training examples
%%% DON'T CHANGE
X = [0 1; 1 1; 1 0; 0 0];
y = [1;0;1;0];

%%% 3. Initialize weight matrices

%%% Number of input neurons
%%% DON'T CHANGE
input_neurons = 2;

%%% Number of hidden layer neurons
%%% This you can change
hidden_neurons = 2;

%%% Number of output layer neurons
%%% DON'T CHANGE
output_neurons = 1;

%%% DON'T CHANGE
% W1 is a 3 x X matrix - 2 + 1 input neurons, X hidden layer neurons
rng(123);
e_init_1 = sqrt(6) / sqrt(input_neurons + hidden_neurons);
W1 = 2*e_init_1*rand(input_neurons + 1,hidden_neurons) - e_init_1;

% W2 is a (X + 1) x 1 matrix - X + 1 hidden layer neurons, 1 output layer neuron
e_init_2 = sqrt(6) / sqrt(hidden_neurons + output_neurons);
W2 = 2*e_init_2*rand(hidden_neurons + 1,output_neurons) - e_init_1;

%%% 4. Repeat k times
%%% DON'T CHANGE
k = 150;

%%% 5. Some relevant variables
%%% DON'T CHANGE
m = size(X,1);
n = size(X,2);

%%% 6. Initialize cost array
%%% DON'T CHANGE
costs = zeros(k,1);

%%% 7. Set learning rate
%%% DON'T CHANGE
alpha = 5;

%%% 8. Implement Stochastic Gradient Descent
%%% PLACE YOUR CODE HERE
for i = 1:k
    epoch_cost = 0;
    for j = 1:m
        % Step 1: Get example as column vector and add bias
        X0 = [1, X(j, :)]';           

        % Step 2: Hidden layer input and activation
        S1 = W1' * X0;                 
        X1 = [1; sigmoid(S1)];        

        % Step 3: Output layer
        S2 = W2' * X1;                 
        X2 = sigmoid(S2);             
        
        %cost of each training example
        y_j = y(j);
        example_cost = -y_j * log(X2) - (1 - y_j) * log(1 - X2);

        epoch_cost = epoch_cost + example_cost;

        %back propagation
        delta2 = (X2 - y(j)) .* dsigmoid(S2);

        Wtilde2 = W2(2:end, :);

        delta1 = dsigmoid(S1) .* (Wtilde2 * delta2);

        grad_W2 =  X1 * delta2';
        grad_W1 =  X0 * delta1';

        %updating weights
        W1 = W1 - (alpha * grad_W1);
        W2 = W2 - (alpha * grad_W2);
    end
    costs(i) = epoch_cost / m;
end

%%% 9. Plot the XOR points as well as the decision regions
%%% PLACE YOUR CODE HERE
plot_XOR_and_regions(W1, W2);
%%% 10. Plot the cost per iteration
%%% PLACE YOUR CODE HERE
figure;
plot(1:k, costs, 'LineWidth', 2);
xlabel('Epoch');
ylabel('Average Cost');
title('Training Cost per Epoch');
grid on;
