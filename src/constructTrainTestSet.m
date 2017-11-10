function constructTrainTestSet(fontNames, trainNum, testNum)
% fontNames: list of font names to work on
% trainNum: number of characters used for training (for each font)
% testNum: number of characters used for testing (for each font)
disp('Begin training')
[folderPath, ~, ~] = fileparts(which(mfilename));

trainSet = struct([]);
testSet = struct([]);
for fontName = fontNames
    disp(strcat('Training data for font: ', fontName))
    data = load(fullfile(folderPath, sprintf('../data/features_%s.mat', fontName)), 'features');
    for index = 1:trainNum
        coord = zeros(20, 1);
        for i = 1:10
            tempCornerPoint = data.features((index - 1) * 10 + i);
            coord(i) = tempCornerPoint.Location(1);
            coord(i + 10) = tempCornerPoint.Location(2);
        end
        trainSet(end + 1).coord = coord;
        trainSet(end).font = fontName;
    end
    
    disp(strcat('Testing data for font: ', fontName))
    for index = (trainNum + 1):(trainNum + testNum)
        coord = zeros(20, 1);
        for i = 1:10
            tempCornerPoint = data.features((index - 1) * 10 + i);
            coord(i) = tempCornerPoint.Location(1);
            coord(i + 10) = tempCornerPoint.Location(2);
        end
        testSet(end + 1).coord = coord;
        testSet(end).font = fontName;
    end
end

assignin('base', 'trainSet', trainSet)
assignin('base', 'testSet', testSet)
save(fullfile(folderPath, '../data/trainTestSet.mat'), 'trainSet', 'testSet');
end