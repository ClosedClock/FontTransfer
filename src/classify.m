function fontIndex = classify(featureVector, k)
trainSet = evalin('base', 'trainSet');

function d = distance(p1, p2)
    d = norm(squeeze(p1 - p2));
end

fontNum = size(trainSet, 1);
trainNum = size(trainSet, 2);
distances = zeros(fontNum, trainNum);
for i = 1:fontNum
    for j = 1:trainNum
        distances(i, j) = distance(featureVector, trainSet(i, j, :));
    end
end
[~, indices] = mink(reshape(distances, [1, numel(distances)]), k);

fontIndex = mode(floor(indices / trainNum) + 1);

end