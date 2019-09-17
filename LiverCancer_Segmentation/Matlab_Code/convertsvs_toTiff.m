clear all;
clc;

%I = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_014/01_01_0101.svs');
%mask_whole = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_014/01_01_0101_whole.tif');
mask_viable1 = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_014/01_01_0101_viable.tif');

mask_viable2 = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_015/01_01_0103_viable.tif');
mask_viable3 = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_001/01_01_0083_viable.tif');


%mask_whole(mask_whole==1)= 255;
%mask_whole = uint8(mask_whole);

mask_viable1(mask_viable1==1)= 255;
mask_viable1 = uint8(mask_viable1);

%figure,imshow(I);
%figure,imshow(mask_whole);

% Display the image1 with grids
figure,imshow(mask_viable1);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable1);
hold on;
for row = 1 : 256 : rows
  line([1, columns], [row, row], 'Color', 'r');
end
for col = 1 : 256 : columns
  line([col, col], [1, rows], 'Color', 'r');
end

figure,imshow(mask_viable1);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable1);
hold on;
for row = 1 : 512 : rows
  line([1, columns], [row, row], 'Color', 'g');
end
for col = 1 : 512 : columns
  line([col, col], [1, rows], 'Color', 'g');
end

figure,imshow(mask_viable1);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable1);
hold on;
for row = 1 : 1024 : rows
  line([1, columns], [row, row], 'Color', 'b');
end
for col = 1 : 1024 : columns
  line([col, col], [1, rows], 'Color', 'b');
end

% Display the image2 with grids

mask_viable2(mask_viable2==1)= 255;
mask_viable2 = uint8(mask_viable2);

%figure,imshow(I);
%figure,imshow(mask_whole);

% Display the image1 with grids
figure,imshow(mask_viable2);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable2);
hold on;
for row = 1 : 256 : rows
  line([1, columns], [row, row], 'Color', 'r');
end
for col = 1 : 256 : columns
  line([col, col], [1, rows], 'Color', 'r');
end

% Display the image1 with grids
figure,imshow(mask_viable2);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable2);
hold on;
for row = 1 : 512 : rows
  line([1, columns], [row, row], 'Color', 'g');
end
for col = 1 : 512 : columns
  line([col, col], [1, rows], 'Color', 'g');
end

% Display the image1 with grids
figure,imshow(mask_viable2);
axis on;
[rows, columns, numberOfColorChannels] = size(mask_viable2);
hold on;
for row = 1 : 1024 : rows
  line([1, columns], [row, row], 'Color', 'b');
end
for col = 1 : 1024 : columns
  line([col, col], [1, rows], 'Color', 'b');
end

%I(repmat(mask_whole,[1,1,3])~=0)= 0;
%I(repmat(mask_viable,[1,1,3])~=0)= 0;


%mask3 = cat(3, mask_viable, mask_viable, mask_viable);
%Im  = I;
%Im(mask3) = 0;


