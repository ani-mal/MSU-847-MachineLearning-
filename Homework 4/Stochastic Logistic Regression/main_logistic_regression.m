load data.txt
load labels.txt 

[m, n] = size(data) 

data = [ ones(m,1) data ] 
labels( labels==0 ) = -1 

test_x = data(2001:4601,:);
test_y = labels(2001:4601);

sampleSize = [200; 500; 800; 1000; 1500; 2000];
acc = [0; 0; 0; 0; 0; 0];

epsilon = 1e-5
maxiter = 1000



function [weights] = logistic_tain(data, labels, epsilon, maxiter)
    %
% code to train a logistic regression classifier
%
% INPUTS:
%   data    = n * (d+1) matrix weightsithn samples and d features, weightshere
%             column d+1 is all ones (corresponding to the intercept term)
%   labels  = n * 1 vector of class labels (taking values 0 or 1)
%   epsilon = optional argument specifying the convergence
%             criterion - if the change in the absolute difference in
%             predictions, from one iteration to the next, averaged across
%             input features, is less than epsilon, then halt
%             (if unspecified, use a default value of 1e-5)
%   maxiter = optional argument that specifies the maximum number of
%             iterations to execute (useful weightshen debugging in case your
%             code is not converging correctly!)
%             (if unspecified can be set to 1000)
%
% OUTPUT:
%   weights = (d+1) * 1 vector of weights weightshere the weights correspond to
%             the columns of "data"

    learning_rate = 0.0005;
    [m_samples, n_features] = size(data) 
    
    weights = zeros(n_features,1);

    old_sig = sigmoid(-data * weights) ;
    for iter = 1:maxiter
        gradient = zeros(n_features,1);
      
        for i = 1:size(data,1)
            sig =  sigmoid(  labels(i) * data(i,:) * weights );
            % update the gradient 
            gradient = gradient + labels(i) * sig * data(i,:)';
        end
        
        % devide the gradient by the number of samples and change it to
        % negative 
        gradient = - gradient / m_samples;
       
        weights = update_weights( weights, learning_rate, gradient )
        
        new_sig = sigmoid(-data * weights);
        if sum(abs(new_sig-old_sig))/m_samples < epsilon
            break;
        end
        
        old_sig = new_sig;
    end
end 

function [weights] = update_weights( prev_weight, learning_rate, gradient )
     weights = prev_weight + learning_rate * (-gradient);
end 

function sig = sigmoid( input )

    sig = 1./ (1 + exp(input));
end

function [prediction] = predict( test_x, weights )
      prediction = test_x* weights
end 

function [acc] = performance(test_x, test_y, weights )
    sum = 0
    [test_col] =  size(test_x, 1)
    prediction =  predict( test_x, weights )
    
    for i = 1:test_col
        if (prediction(i) >= 0 && test_y(i) == 1) || ( prediction(i) < 0 && test_y(i) == -1)
            sum = sum + 1;
        end
    end
    acc = sum / test_col;
end
