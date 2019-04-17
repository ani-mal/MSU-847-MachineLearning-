load data_Spiral.mat

m = mean(D)

Mbar = D - m 

covMat = Mbar*Mbar'

[V, Vec] = eig(covMat)

s = size( D, 1)

K = 3
new_data = V(:, s-K:s)

labels = my_k_means( new_data, K)

gscatter(D(:,1),D(:,2),labels)