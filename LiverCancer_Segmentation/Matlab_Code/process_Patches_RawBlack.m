clear all; clc;

file1 = dir('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/Processed_Raw_Black_Images/patches_test_all/01_01_0091*');
files = file1(1:length(file1));
file = files';
filename = '01_01_0091';

filename1 = strcat('/raid/data/home/matlab/Mousumi/Raw_Images_Black/01_01_0091_viable.tif'); %01_01_0103_viable.tif';
viablemask = imread(filename1);
viablemask(viablemask==1) = 255;
viablemask = uint8(viablemask);
viablemask = imbinarize(viablemask);

fid = fopen( '/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/256x256_Overlap/256_test.csv', 'a' );
k = 1;
train_label_matrix = {};

for idx = 1:length(file)
    % for each WSI listed in set1 do patch extraction
    ifile = file(idx);
    name1 = ifile.name;
    a = imread(strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/Processed_Raw_Black_Images/patches_test_all/', name1));
    [row col dim] = size(a);
    
    part1 = split(name1, '.tiff');
    part2 = split(part1(1), '_');
    
    [viable] = tumor_find(viablemask, str2double(part2(4)), str2double(part2(5)));
    [row col dim] = size(viable);
    
    viable_tumor_Pixels = viable(:,:) ==1 ;
    count_viable = sum(viable_tumor_Pixels(:));
    
    
    %figure,imshow(uint8(a));
    a_gray = rgb2gray(a);
    a_bin = imbinarize(a_gray);
    %figure,imshow(a_gray);
    a_bin = logical(a_bin);
    %figure,imshow(a_bin);
    tissue_region = sum(a_bin(:)==0); % this gives the number of pixels inside the tissue region
    tissue_bg = a(:,:,1) >= 210 & a(:,:,2) >= 210 & a(:,:,3) >= 210 ;
    count_bg = sum(tissue_bg(:));
    
    if ((tissue_region / (1024*1024) > 0.25) && (count_bg /(1024 * 1024) < 0.7) && (row == 1024) && (col == 1024)) 
        
        J =  a;  %block_struct.data;
        %figure,imshow(J);
        patch_name = name1 ; %sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2));
        
        if (count_viable / (1024*1024) < 0.3)
            
            
            train_label_matrix{k,1} = patch_name;
            train_label_matrix{k,2} = 0;     % label 0 : non viable tumor region
            k= k+1;
        else
            train_label_matrix{k,1} = patch_name;
            train_label_matrix{k,2} = 1;     % label 1 : viable tumor region
            k= k+1;
        end
        
        % ------------------------------------------
        % Now process J to extract patches of size 512x512 and 256x256
        
        x_cen = row/2; 
        y_cen = col/2;

        % First extract patch of size 512x512
        xmin_L3 = (x_cen - 256);
        ymin_L3 = (y_cen - 256);
        width_L3 = 511;
        height_L3 = 511;
        J_512 = imcrop(J,[xmin_L3 ymin_L3 width_L3 height_L3]);      
    
        % Extract patch of size 256x256
        xmin_L2 = (x_cen - 128);
        ymin_L2 = (y_cen - 128);
        width_L2 = 255;
        height_L2 = 255;
        J_256 = imcrop(J,[xmin_L2 ymin_L2 width_L2 height_L2]);
        
        % ----------------------------------------------------------
        % Now process viable to extract patches of size 512x512 and 256x256
    
        x_cen = row/2; 
        y_cen = col/2;

        % First extract patch of size 512x512
        xmin_L3 = (x_cen - 256);
        ymin_L3 = (y_cen - 256);
        width_L3 = 511;
        height_L3 = 511;
        viable_512 = imcrop(viable,[xmin_L3 ymin_L3 width_L3 height_L3]);      
    
        % Extract patch of size 256x256
        xmin_L2 = (x_cen - 128);
        ymin_L2 = (y_cen - 128);
        width_L2 = 255;
        height_L2 = 255;
        viable_256 = imcrop(viable,[xmin_L2 ymin_L2 width_L2 height_L2]);
        
        %%%%%%%%%%%%%%%%%%%%%
        % resize 1024x1024 (J, viable) 2 times to get patches of size
        % 256x256 in 5x resolution
        J1 = imresize(J,.5,'Antialiasing',true); 
        J2_5x = imresize(J1,.5,'Antialiasing',true); 
        viable1 = imresize(viable,.5,'Antialiasing',true); 
        viable2_5x = imresize(viable1,.5,'Antialiasing',true); 
        
        %figure,imshow(J2_5x);
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/5x/img/Test/', name1);
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/5x/gt/Test/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), str2double(part2(4)), str2double(part2(5))));
        imwrite(J2_5x,output_filename);
        imwrite(viable2_5x,output_filename1);
        % -----------------------------------------------------
        % resize 512x512 (J_512, viable_512) 1 time to get patches of size
        % 256x256 in 10x resolution      
        
        J1_10x = imresize(J_512,.5,'Antialiasing',true); 
        viable1_10x = imresize(viable_512,.5,'Antialiasing',true); 
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/10x/img/Test/', name1);
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/10x/gt/Test/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), str2double(part2(4)), str2double(part2(5))));
        imwrite(J1_10x,output_filename);
        imwrite(viable1_10x,output_filename1);
        % ------------------------------------------------------
        % save J_256, viable_256 of size 256x256 at 20x resolution 
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/20x/img/Test/', name1);
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/20x/gt/Test/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), str2double(part2(4)), str2double(part2(5))));
        imwrite(J_256,output_filename);
        imwrite(viable_256,output_filename1);

    end
    
    

% for jj = 1 : length( train_label_matrix )
%     fprintf( fid, '%s,%d\n', train_label_matrix(jj,1), train_label_matrix(jj,2));
% end

     
 
  
    

end

for p=1:size(train_label_matrix,1)
         fprintf(fid,'%s,%d\n',train_label_matrix{p,:});
end
fclose(fid);