%% 
% Applying detectFASTFeatures to images and save corners extracted
function extractFeatures_corners(fontFiles, startIndex, endIndex)
% fontFiles: list of font files to work on
% startIndex: first index of the images to extract features from
% endIndex: last index of the images to extract features from

disp('Begin extracting')
[folderPath, ~, ~] = fileparts(which(mfilename));

features = struct([]); % struct list with fields Location and Metric

for fontFile = fontFiles
    fontName = extractBefore(fontFile, '.'); % Get name without extension
    disp(strcat('Extracting features for font: ', fontName))
    for index = startIndex:endIndex
        imageFile = sprintf('%s_%04d.png', fontName, index);
        image = imread(fullfile(folderPath, '../img', char(fontName), imageFile));
        
        % Get corners
        corners = detectFASTFeatures(image, 'MinContrast', 0.1);
        
        % If less than 10, try to find more
        if length(corners) < 10
            corners = detectFASTFeatures(image, 'MinContrast', 0.01);
            disp(['length: ', int2str(length(corners))])
        end
        corners = corners.selectStrongest(10); % Select strongest 10 corners
        for i = 1:10
            ii = (index - startIndex) * 10 + i;
            % If still less than 10, make trivial corners
            if i > length(corners)
                features(ii).Location = [0 0];
                features(ii).Metric = 0;
            else
                features(ii).Location = corners(i).Location;
                features(ii).Metric = corners(i).Metric;
            end
        end
    end
    save(fullfile(folderPath, sprintf('../data/features_%s.mat', fontName)), 'features');
end