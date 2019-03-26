load data.txt
load labels.txt


acc200 = trainAccuracy( 200, data, labels )
acc500 = trainAccuracy( 500, data, labels )
acc800 = trainAccuracy( 800, data, labels )
acc1000 =  trainAccuracy( 800, data, labels )
acc1500 =  trainAccuracy( 800, data, labels )
acc2000 =  trainAccuracy( 800, data, labels )


%w = [-0.0532426211202419,-0.101494299441627,-0.0231653492598399,-0.00311930447392373,1.80368271198996,0.0662435761398366,0.00217748644391415,0.161957061892909,0.0818197302622441,0.000228918844915121,-0.00753525695250824,-0.0371780875971013,-0.0277262183377819,-0.133177239263655,-0.103153968198195,0.165856246132869,0.0885310566454737,0.0515954438555393,0.00995525670278365,0.0223456596798561,-0.00691269926936171,0.0380849617203521,0.442040229025566,0.00784002282518719,0.112381709876713,0.520507138671868,-0.00596514151266454,0.490088441776622,-0.669206889777852,-0.355025743904382,-0.136145394080695,0.556297214958747,-1.17936858847057,0.878143500146321,0.246341500582391,0.905103873102487,-3.49209470988015,0.415834635247503,3.13870142613442,-0.0470439628305015,-0.257492594610768,0.0661419926718814,0.906845632706671,-0.262158364186316,0.696387208962131,-0.150105650561462,1.65341065210332,3.46523657179254,1.08760342218631,0.0704263940281521,-0.0109753979483244,-0.796503400196370,0.0697467678920899,0.112977183750852,-0.0725875584401685,0.0717680451382590,0.0293292646080507,0.00609538398283847]
%acc = predict( w, test, testLabels )

function acc = trainAccuracy( n, data, labels )
    train = data(1:n, :)
    trainLabels = labels(1:n)

    testLabels = labels(n:end)
    test = data(n:end, :)

    epsilon = 0.001 
    maxiter = 1000

    [weights] = logistic_tain(train, trainLabels, epsilon, maxiter)
    [acc] = predict( weights, test, testLabels )
end 

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
    sig = 1./(ones(m,1) + exp((wData) ) )
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
    iter = 0 
    for i= 1:maxiter 
        prevW =  weights
        
        sig = sigmoid( weights, biasData )
        like = likelihood( sig )
        grad = gradient( biasData, like, labels)
       
        lX = like.*biasData
        prevW =  weights
        % weights <-  weights + (biasData'lX)^(-1) * grad
        hess = ( biasData'*lX) %58x58 matrix, we only need its diagonal 
        cost = hess\grad
        weights = prevW - 0.01*cost'
        
        difference = weights' - prevW'
        sumDif = sum( difference )
        dist =  abs(sumDif/ 58)
        
        
        if( dist <= epsilon ) %stop condition
            
            break
        end
    end 
end 


function [accuracy] = predict( weights, test, expected )
    [m,n] = size(test)
    prediction = zeros(1, n)
    
    testBias = [ ones(m,1) test]
    correct = 0
    for i = 1:n 
        result = dot(weights',testBias(i, :))
        if( result > 0.5 )
            prediction(i) = 1      
        end
        if( prediction(i) == expected(i) )
                correct = correct + 1 
        end
    end 
    
    accuracy = (correct/m) *100
end 





