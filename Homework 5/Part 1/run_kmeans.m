load data_Spiral.mat

labels = my_k_means(D, 3)

gscatter(D(:,1),D(:,2),labels)