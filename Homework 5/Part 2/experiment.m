load USPS.mat

A1 = reshape(A(1,:), 16, 16)'
A2 = reshape(A(2,:), 16, 16)'


imwrite(A1, "img1.png")
imwrite(A2, "img2.png")

p = [10, 50, 100, 200]
err = zeros(4,1)

for i=1:4
    coefficient =  pca(A, 'NumComponents', p(i))
    img = (A*coefficient)*coefficient.'
    err(i) = norm(A-img, 'fro')^2
    
    img1 = reshape(img(1,:), 16, 16)'
    img2 = reshape(img(2,:), 16, 16)'
    
    imwrite(img1, p(i) + "img1.png")
    imwrite(img2, p(i) + "img2.png")
end 