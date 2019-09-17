clear all; clc;

file1 = dir('/raid/data/home/matlab/Mousumi/Prediction_Images_UNet/01_01_0113/*');
files = file1(3:length(file1));
file = files';

for i = 1:length(file)
    
    % for each WSI listed in set1 do patch extraction
    ifile = file(i);
    name1 = ifile.name;
    I = imread(strcat('/raid/data/home/matlab/Mousumi/Prediction_Images_UNet/01_01_0113/', name1));
    J = imresize(I, [512 512]);
    J(J==1) = 255;
    imwrite(J, strcat('/raid/data/home/matlab/Mousumi/Prediction_Images_UNet/01_01_0113_resized/',name1));
end

