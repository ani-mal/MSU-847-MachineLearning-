load data.txt
load labels.txt

epsilon = 0.1 
maxiter = 100

train = data(1:2000, :)
trainLabels = labels(1:2000)

testLabels = labels(2000:4601)
test = data(2001:4601, :)


 
%weights =  logistic_tain(train, trainLabels, epsilon, maxiter)
%prediction = predict( weights, test, testLabels )
[m,n] = size(train)
    dataBias = [ones(m,1) train]
    [m,n] = size(dataBias) 
    wieghts = zeros(n, 1)
[th, J_hist] = gradientDescentMulti(dataBias, trainLabels, weights, 0.01, 10)

function [th, J_hist] = gradientDescentMulti(X, y, theta, alpha, num_iters)
     m = length(y); % number of training examples
     J_history = zeros(num_iters, 1);

    for iter = 1:num_iters
         theta = theta -((1/m) * ((X * theta) - y)' * X)' * alpha;
    end
    th=theta;
    J_hist=J_history;
end   
 

function [dataBias, wieghts] = initializeData( data )
    [m,n] = size(data)
    dataBias = [ones(m,1) data]
    [m,n] = size(dataBias) 
    wieghts = zeros(n, 1)
end 

function [weights] = logistic_tain(data, labels, epsilon, maxiter)
%
% code to train a logistic regression classifier
%
% INPUTS:
% data = n * (d+1) matrix withn samples and d features, where
% column d+1 is all ones (corresponding to the intercept term)
% labels = n * 1 vector of class labels (taking values 0 or 1)
% epsilon = optional argument specifying the convergence
% criterion - if the change in the absolute difference in
% predictions, from one iteration to the next, averaged across
% input features, is less than epsilon, then halt
% (if unspecified, use a default value of 1e-5)
% maxiter = optional argument that specifies the maximum number of
% iterations to execute (useful when debugging in case your
% code is not converging correctly!)
% (if unspecified can be set to 1000)
%
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%

% output = b0 + b1*x1 + b2*x2
%then the output is transformed into using a logistic functon
% p(class=0) = 1 / (1 + e^(-output))

[dataBias, weights] = initializeData( data )
[m, n] = size(dataBias)
 for i=1:maxiter
    row = dataBias(j,:)
    prediction = 1/(1 + exp( -(dot(weights,dataBias(j,:)') ) ) )
    distance = labels(j) - prediction
    constant = epsilon*(distance)*prediction*(1 - prediction)
    vector = (dataBias(j,:)) 
    weights = weights + constant.*vector'
 end 
end 
% we start with all the coefficients being zero 
% b0 is the bias 
% b0=0, b1=0, .. bn=0
%then we calculate the new coeficients by updating b
% b = b + epsilon * (y – prediction) * prediction * (1 – prediction) * x

function [accuracy] = predict( weights, test, expected )
    
    [m,n] = size(test)
    prediction = zeros(1, n)
    
    testBias = [ ones(m,1) test]
    correct = 0
    for i = 1:n 
        result = sum(weights'.*testBias(i, :))
        if( result > 0.5 )
            prediction(i) = 1 
            if( prediction(i) == expected(i) )
                correct = correct + 1 
            end
        end
    end 
    
    accuracy = (correct/m) * 100
end 

