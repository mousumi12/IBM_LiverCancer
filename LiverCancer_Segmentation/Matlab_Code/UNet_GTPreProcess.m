clear all; clc;

file1 = dir('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Val/*.tiff');
files = file1(1:length(file1));
file = files';

for idx = 1:length(file)
    
    ifile = file(idx);
    name1 = ifile.name;
    I = imread(strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Val/', name1));
    %figure,imshow(I);
    J = imcomplement(I);
    %figure,imshow(J);
    J = bwareaopen(J, 50);
    %figure,imshow(J);
    K = imcomplement(J);
    %figure,imshow(K);
    imwrite(K, strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor_New/Val/',name1));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file2 = dir('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Train/*.tiff');
files2 = file2(1:length(file2));
file2 = files2';

for idx = 1:length(file2)
    
    ifile = file2(idx);
    name1 = ifile.name;
    I = imread(strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Train/', name1));
    %figure,imshow(I);
    J = imcomplement(I);
    %figure,imshow(J);
    J = bwareaopen(J, 50);
    %figure,imshow(J);
    K = imcomplement(J);
    %figure,imshow(K);
    imwrite(K, strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor_New/Train/',name1));
end


%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
file3 = dir('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Test/*.tiff');
files3 = file3(1:length(file3));
file3 = files3';

for idx = 1:length(file3)
    
    ifile = file3(idx);
    name1 = ifile.name;
    I = imread(strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor/Test/', name1));
    %figure,imshow(I);
    J = imcomplement(I);
    %figure,imshow(J);
    J = bwareaopen(J, 50);
    %figure,imshow(J);
    K = imcomplement(J);
    %figure,imshow(K);
    imwrite(K, strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/dataset_Unet/256_ViableTumor_New/Test/',name1));
end

