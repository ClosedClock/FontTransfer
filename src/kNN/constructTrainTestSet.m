%% 
% Read saved corner datas and pick proper features to construct train
% and test sets.
function constructTrainTestSet(fontNames, trainNum, testNum)
% fontNames: list of font names to work on
% trainNum: number of characters used for training (for each font)
% testNum: number of characters used for testing (for each font)

disp('Begin training')
[folderPath, ~, ~] = fileparts(which(mfilename));

trainSet = zeros(length(fontNames), trainNum, 20);
testSet = zeros(length(fontNames), testNum, 20);
for i = 1:length(fontNames)
    fontName = fontNames(i);
    % Get saved data of corners
    data = load(fullfile(folderPath, sprintf('../data/features_%s.mat', fontName)), 'features');
    % Use Location of corners as features
    for j = 1:trainNum
        for k = 1:10
            trainSet(i, j, k) = data.features(10*j - 10 + k).Location(1);
            trainSet(i, j, k+10) = data.features(10*j - 10 + k).Location(2);
        end
    end
    for j = 1:testNum
        index = j + trainNum;
        for k = 1:10
            testSet(i, j, k) = data.features(10*index - 10 + k).Location(1);
            testSet(i, j, k+10) = data.features(10*index - 10 + k).Location(2);
        end
    end
end

% Save to workspace
assignin('base', 'trainSet', trainSet)
assignin('base', 'testSet', testSet)
% Save to file
save(fullfile(folderPath, '../data/trainTestSet.mat'), 'trainSet', 'testSet');
end