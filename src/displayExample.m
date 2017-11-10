% Take an example image and display its corners

imageFile = 'simkai/simkai_0200.png';

image = imread(fullfile(imageFile));

corners = detectFASTFeatures(image,'MinContrast',0.1);
corners = corners.selectStrongest(10); % Select strongest 10 corners

displayImage = insertMarker(image, corners, 'circle');
figure;
imshow(displayImage);