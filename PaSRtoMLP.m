DD = dlmread('out'); % read the raw PaSR output

S = size(DD);

Nsamples = S(1)/8; % calculate the number of samples

data = zeros(Nsamples,22); 

for ii = 1:Nsamples

    dd = reshape(DD([1:4]+(ii-1)*8, :)',1,12);

    data(ii,1:11) = dd(1:11); %populate the 11 x-entries

    dd = reshape(DD([5:8]+(ii-1)*8, :)',1,12);

    data(ii,12:22) = dd(1:11); % populate the 11 f(x) entries

end

I = randperm(Nsamples);

data = data(I,:); % randomly permute the samples

D = data;

N = 40000; % MLP training will be done on 40000 samples

ndim = 11; % dimension of x

lmin = 0.025; % minimum x-distance between two points in the MLP dataset

near = zeros(N,10); % work array

iPick = [1]; % always pick the first entry in the permuted data

count = 1; % number of samples picked so far

ii = 1; % number of samples considered so far

while count < N

    ii = ii+1; % index of the next point to be considered

    dist = (sum( (D(iPick,[1:ndim]) - D(ii,[1:ndim])).^2, 2)).^0.5; % calculate the distance between the 
    % current point and all other points which have been chosen

    if ( min( dist ) > lmin ) % if the minimum distance is above the threshold, add the current point to the list 
        iPick = [iPick, ii];
        count = count + 1;
    end

end

D = D(iPick,:); % keep only the 40000 points which are spread out from each other

dlmwrite('data.csv',D,'delimiter',',','precision','%.15e'); % output the data to data.csv