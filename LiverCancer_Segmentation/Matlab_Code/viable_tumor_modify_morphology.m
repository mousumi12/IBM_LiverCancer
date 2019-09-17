clear all; clc;

I = imread('/raid/data/home/matlab/Mousumi/Raw_Images/01_01_0113.svs');
%figure,imshow(I);
J = imread('/raid/data/home/matlab/Mousumi/viable_tumors/01_01_0113_viable.tif');
J(J==1) = 255;

[row col dim] = size(J);
%J1 = imfill(J,'holes');
J = imcomplement(J);
%figure,imshow(J);

J = bwareaopen(J, 5000);

figure,imshow(J);
%figure,imshow(J);



