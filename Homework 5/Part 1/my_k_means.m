function [clusters_index] = my_k_means(data, K)

    centroids = initialize_centroids(data, K);
    m = size(data, 2)
    oldcentroids = zeros(K, m)
    interations = 0 

    while oldcentroids~= centroids
      oldcentroids = centroids
      cluster_number = getClosestCentroids(data, centroids);
      centroids = calculate_centroids(data, cluster_number, K);
      interations = interations + 1 
    end

    final = [data,cluster_number] 

    m = size(final, 1) 

    clusters_index =  cluster_number;

   % gscatter(final(:,1),final(:,2),final(:,3))
%     plot(centroids(1,1), centroids(1,2), 'k.','MarkerSize',20)
%     plot(centroids(2,1), centroids(2,2), 'k.','MarkerSize',20)
%     plot(centroids(3,1), centroids(3,2), 'k.','MarkerSize',20)
%     plot(centroids(4,1), centroids(4,2), 'k.','MarkerSize',20)
end 

function centroids = initialize_centroids(data, K)
    centroids = zeros(K,size(data,2)); 
    randidx = randperm(size(data,1));
    centroids = data(randidx(1:K), :);
 end
  
 function cluster_number = getClosestCentroids(data, centroids)
  K = size(centroids, 1);
  cluster_number = zeros(size(data,1), 1);
  m = size(data,1);

  for i=1:m
    k = 1;
    min_dist = sum((data(i,:) - centroids(1,:)) .^ 2);
    for j=2:K
        dist = sum((data(i,:) - centroids(j,:)) .^ 2);
        if(dist < min_dist)
          min_dist = dist;
          k = j;
        end
    end
    cluster_number(i) = k;
  end
 end


function centroids = calculate_centroids(data, idx, K)
  [m n] = size(data);
  centroids = zeros(K, n);
  
  for i=1:K
    xi = data(idx==i,:);
    ck = size(xi,1);
    centroids(i, :) = (1/ck) * sum(xi);
  end
end



