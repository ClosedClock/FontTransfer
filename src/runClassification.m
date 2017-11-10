function result = runClassification(fontNames, k)
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

result = zeros(length(fontNames));
% for i = 1:length(testSet)
%     actualFont = testSet(i).font;
%     disp(strcat("actualFont: ", actualFont))
%     actualIndex = find(fontNames == actualFont);
%     classifiedFont = classify(testSet(i), k);
%     disp(strcat("classifiedFont: ", classifiedFont))
%     classifiedIndex = find(fontNames == classifiedFont);
%     
%     result(classifiedIndex, actualIndex) = result(classifiedIndex, actualIndex) + 1;
% end
% result = result / length(testSet) * length(fontNames);

for i = 1:size(testSet, 1)
    for j = 1:size(testSet, 2)
        output = classify(testSet(i, j, :), k);
        result(output, i) = result(output, i) + 1;
    end
end
result = result / size(testSet, 2);

end
