clear all; clc;



secondlevelTIFF = imread('/raid/data/home/matlab/Mousumi/Raw_Images/01_01_0096.svs');
imshow(secondlevelTIFF);
axis on;
[rows, columns, numberOfColorChannels] = size(secondlevelTIFF);
hold on;
for row = 1 : 512 : rows
  line([1, columns], [row, row], 'Color', 'g');
end
for col = 1 : 512 : columns
  line([col, col], [1, rows], 'Color', 'g');
end