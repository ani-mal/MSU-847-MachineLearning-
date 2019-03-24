load data.txt
load labels.txt

epsilon = 0.1 
maxiter = 100

train = data(1:2000, :)
trainLabels = labels(1:2000)

testLabels = labels(2000:4601)
test = data(2001:4601, :)


function sig = sigmoid( weights, vector )
    sig =  = 1/(1 + exp( -(dot(weights,vector') ) ) )
end 

function l = log_likelihood(weights, vector)

    sig_prob = sigmoid( weights, vector)
end 