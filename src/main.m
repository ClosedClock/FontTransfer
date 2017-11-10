fontFiles = ["simkai.ttf", "Xingkai.ttc", "Baoli.ttc", "Songti.ttc"];
fontNames = extractBefore(fontFiles, '.');
%extractFeatures(fontFiles, 100, 999);

%constructTrainTestSet(fontNames, 700, 200);

%extractFeatures_with_bag_of_feature(fontNames, 50, 20);

if false
firstIndex = 5;
secondIndex = 10;
figure;
hold on;
scatter(trainSet(1, :, firstIndex), trainSet(1, :, secondIndex), 'b');
scatter(trainSet(2, :, firstIndex), trainSet(2, :, secondIndex), 'g');
scatter(trainSet(3, :, firstIndex), trainSet(3, :, secondIndex), 'r');
scatter(trainSet(4, :, firstIndex), trainSet(4, :, secondIndex), 'm');
title('trainSet')

figure;
hold on;
scatter(testSet(1, :, firstIndex), testSet(1, :, secondIndex), 'b');
scatter(testSet(2, :, firstIndex), testSet(2, :, secondIndex), 'g');
scatter(testSet(3, :, firstIndex), testSet(3, :, secondIndex), 'r');
scatter(testSet(4, :, firstIndex), testSet(4, :, secondIndex), 'm');
title('testSet')
end

result = runClassification(fontNames, 10);
disp(result)