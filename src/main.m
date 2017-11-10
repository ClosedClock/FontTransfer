fontFiles = ["simkai.ttf", "Xingkai.ttc", "Baoli.ttc", "Songti.ttc"];
fontNames = extractBefore(fontFiles, '.');
%extractFeatures(fontFiles, 100, 999);

%constructTrainTestSet(fontNames, 700, 200);

extractFeatures_with_bag_of_feature(fontNames, 500, 200);

result = runClassification(fontNames, 3);
disp(result)