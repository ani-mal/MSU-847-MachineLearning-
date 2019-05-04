img = imread('flat.bmp')
X = double(img)
channel1 = X(:,:,1) % get first channel
channel2 = X(:,:,2) % get second channel 
channel3 = X(:,:,3) % get third channel 

rankVal = [1, 5, 10, 15, 20, 25, 30]
 
[compleChannel1, missing1, err1] =  computeCompletion( channel1, rankVal )
[compleChannel2, missing2, err2] =  computeCompletion( channel2, rankVal )
[compleChannel3, missing3, err3] =  computeCompletion( channel3, rankVal )

constructed = cat(3, compleChannel1, compleChannel2, compleChannel3)

%% summing the error from all channels 
errors = [ err1';err2';err3']
errors = sum(errors)

%%
% plot errors
figure
plot(rankVal, errors, 'x--')
xlabel('Hard Impute with Rank r')
ylabel('Recovery Errors')

%% Plotting the images 
figure
hold on
ax = subplot(3,3,1)
set ( ax, 'visible', 'off')
ax = subplot(3,3,1)
imshow(uint8(X))
title('Original Colored Image')

ax = subplot(3,3,2)
set ( ax, 'visible', 'off')
missing =  cat(3, missing1, missing2, missing3)
imshow(uint8(missing))
title('Noise Colored Image')

for i=1:length(rankVal)
    ax = subplot(3,3,i+2)
    set ( ax, 'visible', 'off')
    im = cat(3, compleChannel1{i}, compleChannel2{i}, compleChannel3{i})
    imshow(uint8(im))
    title(['r=' num2str(rankVal(i))])
end

hold off;


function [completedMatrices, X_missing, error] = computeCompletion( X, rankVal)

 %% resizing image 
    idx = randperm(128*128,128*128/2)
    X_missing = X
    X_missing(idx) = 0
    Omega = true(128,128)
    Omega(idx) = false
%%
   
    len = length(rankVal)

%% Applying har impute 
    completedMatrices = {}
    error = zeros(len,1)
    for i=1:len
        completedMatrices{i} = hardimpute(X_missing, Omega, rankVal(i))
        error(i) = sum(sum((completedMatrices{i}-X).^2))
    end

end

