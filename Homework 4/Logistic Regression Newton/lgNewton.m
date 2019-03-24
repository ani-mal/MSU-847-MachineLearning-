load data.txt
load labels.txt

epsilon = 0.1 
maxiter = 100

train = data(1:2000, :)
trainLabels = labels(1:2000)

testLabels = labels(2000:4601)
test = data(2001:4601, :)

[m,n] = size(data)

weights = zeros(1, n+1)

[m,n] = size(train)
train = [ ones(m,1) train ]

sig = sigmoid( weights, train )
l = likelihood( sig )

function sig = sigmoid( weights, data )
    [m,n] = size(data)
    tdata = data'
    wData = data*weights'
    sig = 1./(ones(m,1) + exp( -(wData) ) )
end 


function l = likelihood( sig )
   [m,n] = size(sig)

    l = sig.*( ones(m,1) - sig)
end






