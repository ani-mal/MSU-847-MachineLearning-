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