%%
% Return the classified font index for featureVector
function fontIndex = classify(featureVector, k)
% featureVector: feature vector, should be a double array
% k: k nearest neighbors

% Load train set from workspace
trainSet = evalin('base', 'trainSet');

% Define distance function. Here it is Euclidean
function d = distance(p1, p2)
    d = norm(squeeze(p1 - p2));
end

fontNum = size(trainSet, 1);
trainNum = size(trainSet, 2);
distances = zeros(fontNum, trainNum);

% Calculate distances
for i = 1:fontNum
    for j = 1:trainNum
        distances(i, j) = distance(featureVector, trainSet(i, j, :));
    end
end

% Find indices of k min values
[~, indices] = mink(reshape(distances, [1, numel(distances)]), k);

% Find most common one (mode)
fontIndex = mode(floor(mod(indices - 1, fontNum)) + 1);

end