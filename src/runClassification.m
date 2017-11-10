%%
% Do kNN classification with train and test sets in workspace
function result = runClassification(fontNames, k)
% fontNames: list of font names to work on
% k: classify according to k nearest neighbors

% If there is no data in workspace, try to load from file
if ~ismember('trainSet',evalin('base','who')) || ~ismember('trainSet',evalin('base','who'))
    [folderPath, ~, ~] = fileparts(which(mfilename));
    disp('trainSet or testSet not exist. Try to load from files.')
    data = load(fullfile(folderPath, '../data/trainTestSet.mat'));
    testSet = data.testSet;
    trainSet = data.trainSet;
    assignin('base', 'trainSet', trainSet)
else
    testSet = evalin('base', 'testSet');
end

% result(i, j) is the probability that j-th font is classified as i-th font
result = zeros(length(fontNames));

% Do classification for test set
for i = 1:size(testSet, 1)
    for j = 1:size(testSet, 2)
        output = classify(testSet(i, j, :), k);
        result(output, i) = result(output, i) + 1;
    end
end

result = result / size(testSet, 2);
end
