clear all; clc;

% First do for images

str1 = 'J:\Necrosis_TCGA\Patches\2048x2048\Images\';
files1 = dir(str1);
file1 = files1';

for ifile = file1(3:length(file1))
    
    name1 = ifile.name;
    I = imread(strcat(str1,name1));
    [row, col, dim] = size(I);
    
    if row == 2048 && col == 2048
        x_cen = row/2; 
        y_cen = col/2;

        % First extract patch for L3
        xmin_L3 = (x_cen - 512);
        ymin_L3 = (y_cen - 512);
        width_L3 = 1023;
        height_L3 = 1023;
        J = imcrop(I,[xmin_L3 ymin_L3 width_L3 height_L3]);
    
        %new_name3 = strrep(name1,'L4','L3');
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\1024x1024\Images\',name1));
    
        % Extract patch for L2
        xmin_L2 = (x_cen - 256);
        ymin_L2 = (y_cen - 256);
        width_L2 = 511;
        height_L2 = 511;
        J = imcrop(I,[xmin_L2 ymin_L2 width_L2 height_L2]);
    
        
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\512x512\Images\',name1));
    
    
        % Extract patch for L1
        xmin_L1 = (x_cen - 128);
        ymin_L1 = (y_cen - 128);
        width_L1 = 255;
        height_L1 = 255;
        J = imcrop(I,[xmin_L1 ymin_L1 width_L1 height_L1]);
    
        
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\256x256\Images\', name1));
    
    else
        continue;
    end
        
end

% NOw do for mask images/ Labels

str2 = 'J:\Necrosis_TCGA\Patches\2048x2048\Label\';
files2 = dir(str2);
file2 = files2';

for ifile = file2(3:length(file2))
    
    name1 = ifile.name;
    I = imread(strcat(str2,name1));
    [row, col, dim] = size(I);
    
    if row == 2048 && col == 2048
        x_cen = row/2; 
        y_cen = col/2;

        % First extract patch for L3
        xmin_L3 = (x_cen - 512);
        ymin_L3 = (y_cen - 512);
        width_L3 = 1023;
        height_L3 = 1023;
        J = imcrop(I,[xmin_L3 ymin_L3 width_L3 height_L3]);
    
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\1024x1024\Label\',name1));
    
        % Extract patch for L2
        xmin_L2 = (x_cen - 256);
        ymin_L2 = (y_cen - 256);
        width_L2 = 511;
        height_L2 = 511;
        J = imcrop(I,[xmin_L2 ymin_L2 width_L2 height_L2]);
    
        
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\512x512\Label\',name1));
    
    
        % Extract patch for L1
        xmin_L1 = (x_cen - 128);
        ymin_L1 = (y_cen - 128);
        width_L1 = 255;
        height_L1 = 255;
        J = imcrop(I,[xmin_L1 ymin_L1 width_L1 height_L1]);
    
        
        imwrite(J, strcat('J:\Necrosis_TCGA\Patches\256x256\Label\', name1));
    
    else
        continue;
    end
        
end


