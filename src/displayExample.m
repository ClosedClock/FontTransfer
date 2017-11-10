imageFile = 'simkai_0006.png';

image = imread(fullfile(imageFile));

corners = detectFASTFeatures(image,'MinContrast',0.001);
corners = corners.selectStrongest(10); % Select strongest 10 corners

displayImage = insertMarker(image, corners, 'circle');
figure;
imshow(displayImage);