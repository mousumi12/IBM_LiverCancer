% Take 1024x1024 with overlap 1024x1024 then total size will be 2048x2048

clear all; clc;
    global filename
    global filename1
    global viablemask
    global filename2
    global wholemask

files = dir('/raid/data/home/matlab/Mousumi/Raw_Images/*.svs');
file = files';


for i = 1:1 %length(file)
    % for each WSI listed in set1 do patch extraction
    ifile = file(i);
    name1 = ifile.name;
    % Read the svs file for highest resolution corresponds to Index = 1
    
    tileSize = [512, 512];

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
    [~,baseFilename,~]=fileparts(input_svs_file);

    filename1 = '/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_015/01_01_0103_viable.tif';
    viablemask = imread(filename1);
    viablemask(viablemask==1) = 255;
    viablemask = uint8(viablemask);
    viablemask = imbinarize(viablemask);
    %figure, imshow(viablemask);
    
    %filename2 = 'K:\PAIP_Challenge\All\Done\Training_phase_1_001\01_01_0083_whole.tif';
    %wholemask = imread(filename2);
    %wholemask(wholemask==1) = 255;
    %wholemask = uint8(wholemask);
    %wholemask = imbinarize(wholemask);
    %figure, imshow(wholemask);
    
    filename = baseFilename;
    %disp filename
    svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page); %create an adapter which modulates how the large svs file is accesse
    tic
    %fun =@(block)feature_extraction(block.data);
   
    % Non-overlapping
    blockproc(svs_adapter,tileSize,@fun)     
    toc 
end 
    
    % Check if the 70% of teh image area is only background remove that patch,
    % generate the patch and write in disk
    
function fun(block_struct)
    
    global filename
    global filename1
    global viablemask
    global filename2
    global wholemask

    a = block_struct.data;
    [row col dim] = size(a);
    
    [viable] = tumor_find(viablemask ,block_struct.location(1),block_struct.location(2));
    [row col dim] = size(viable);
    
    viable_tumor_Pixels = viable(:,:) ==1 ;
    count_viable = sum(viable_tumor_Pixels(:));
    
    %[whole] = tumor_find(wholemask,block_struct.location(1),block_struct.location(2));
    %[row col dim] = size(whole);
    
    %whole_tumor_Pixels = whole(:,:) ==1 ;
    %count_whole = sum(whole_tumor_Pixels(:));
        
    %figure,imshow(uint8(a));
    a_gray = rgb2gray(a);
    a_bin = imbinarize(a_gray);
    %figure,imshow(a_gray);
    a_bin = logical(a_bin);
    %figure,imshow(a_bin);
    tissue_region = sum(a_bin(:)==0); % this gives the number of pixels inside the tissue region
    tissue_bg = a(:,:,1) >= 210 & a(:,:,2) >= 210 & a(:,:,3) >= 210 ;
    count_bg = sum(tissue_bg(:));
    
    
    
    
    % carefully fix values here
    
    if ((tissue_region / (512*512) > 0.25) && (count_bg /(512 * 512) < 0.7) && (row == 512) && (col == 512))    %check the ratio of total no of pixels in [512 * 512] image , to remove the background patches.     
        
        %Resize the patch to get patch of size 1024x1024
        J =  block_struct.data;
        
        %if (count_whole / (512*512) <0.2
        
        
        
        J1 = imresize(J,.5,'Antialiasing',true); 
        J2 = imresize(viable,.5,'Antialiasing',true); 
        
        %output_filename1 = strcat('K:\PAIP_Challenge\dataset\256\train\',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        output_filename2 = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_viable_masks/',sprintf('%s_%d_%d.tiff','viable', block_struct.location(1),block_struct.location(2)));
        %imwrite(J1,output_filename1);
        imshow(J2);
        imwrite(J2,output_filename2);
        
        
        
    end
end
