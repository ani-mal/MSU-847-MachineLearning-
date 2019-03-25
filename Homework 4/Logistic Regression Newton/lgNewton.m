load data.txt
load labels.txt

train = data(1:2000, :)
trainLabels = labels(1:2000)

testLabels = labels(2000:4601)
test = data(2001:4601, :)

[weights, biasData] = initialize(train)

 sig = sigmoid( weights, biasData )
%  p = likelihood( sig )
%  lx = p.*biasData
%  l = gradient( biasData, p, trainLabels ) 
%  h = hessian( weights, biasData)
%  R = diag(h)
 
%  result = biasData.*R
%  rGrad = result\l
%  
epsilon = 0.0001 
maxiter = 1000

[weights] = logistic_tain(train, trainLabels, epsilon, maxiter)

%initializing weights as zero, and adding the biad column of 1 to the data
%points 
function [weights, biasdata] = initialize( data )
    [m,n] = size(data)
    weights = zeros(1, n+1)
    biasdata = [ ones(m,1) data ]
end 

%sigmoid function 
% 1/(1 + exp(-z) 
function sig = sigmoid( weights, data )
    [m,n] = size(data)
    wData = data*weights'
    sig = 1./(ones(m,1) + exp( -(wData) ) )
end 

%Max likelihood sigma(1- sigma)
function p = likelihood( sig )
   [m,n] = size(sig)
    p = sig.*( ones(m,1) - sig)
end

% phi(y-t)
function l = gradient( data,y, labels)
    l = data'*(y - labels ) 
end 


%Rnn = y_n(1-y_n) = w^t phi_n( 1 - w^t phi )
% returns a vector ?
function h = hessian( weights, data)
    [m,n] = size(data)
    wPhi =  weights*data' %yn = w^t * phi 
    h = wPhi.*( ones(1,m) - wPhi)  
end 

function [weights] = logistic_tain(data, labels, epsilon, maxiter)
% code to train a logistic regression classifier
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
% OUTPUT:
% weights = (d+1) * 1 vector of weights where the weights correspond to
% the columns of "data"
%

% output = b0 + b1*x1 + b2*x2
%then the output is transformed into using a logistic functon
% p(class=0) = 1 / (1 + e^(-output))

[weights, biasData] =  initialize( data )

    for i= 1:maxiter 
        prevW =  weights
        
        sig = sigmoid( weights, biasData )
        like = likelihood( sig )
        grad = gradient( biasData, like, labels)
       
        lX = grad'.*biasData
        prevW =  weights
        % weights <-  weights + (biasData'lX)^(-1) * grad
        cost = ( biasData'*lX)\grad
        weights = prevW + cost'
        
        difference = weights' - prevW'
        sumDif = sum( difference )
        dist =  abs(sumDif/ 58)
        
        if( dist <= epsilon ) %stop condition
            break
        end
    end 
end 







