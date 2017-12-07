function extractFeatures_with_bag_of_feature(fontNames, trainNum, testNum)
%clear all;
%setDir  = fullfile(toolboxdir('vision'),'visiondata','imageSets');
[folderPath, ~, ~] = fileparts(which(mfilename));
setDir = fullfile(folderPath, '../img');
imgSets = imageSet(setDir, 'recursive');
%% 
% Pick the first two images from each image set to create training sets.
[trainingSets,TestSets] = partition(imgSets, [50,20]);
%%
% Create the bag of features. This process can take a few minutes.
bag = bagOfFeatures(trainingSets,'Verbose',true,'VocabularySize',50);
%%
%Train a classifier with the training sets.
%categoryClassifier = trainImageCategoryClassifier(trainingSets,bag);
%%
%Evaluate the classifier using test images. Display the confusion matrix.
%confMatrix = evaluate(categoryClassifier,TestSets);
%%
%Find the average accuracy of the classification.
%mean(diag(confMatrix));
%%
%get feature vector
trainSet = zeros(length(fontNames), trainNum, 50);
testSet = zeros(length(fontNames), testNum, 50);
for i = 1:length(fontNames)
    for j = 1:trainNum
        img = read(imgSets(i), j);
        trainSet(i, j, :) = encode(bag, img);
    end
    
    for j = 1:testNum
        img = read(imgSets(i), j + trainNum);
        testSet(i, j, :) = encode(bag, img);
    end
end

% Save to workspace
assignin('base', 'trainSet', trainSet)
assignin('base', 'testSet', testSet)
% Save to file
save(fullfile(folderPath, '../data/trainTestSet.mat'), 'trainSet', 'testSet');
%%
%Apply the newly trained classifier to categorize new images.
% img = imread(fullfile(setDir,'cups','bigMug.jpg'));
% [labelIdx, score] = predict(categoryClassifier,img);
% Display the classification label.
% 
% categoryClassifier.Labels(labelIdx)
end