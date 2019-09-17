clear all; clc;

global filename
global filename1
global viablemask
     
     
name1 = '01_01_0103.svs';
k = 1;
    tileSize = [512, 512];

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/01_01_0103.svs');
    [~,baseFilename,~]=fileparts(input_svs_file);

    filename1 = strcat('/raid/data/home/matlab/Mousumi/viable_tumors/', strtok(name1,'.svs'),'_viable.tif'); %01_01_0103_viable.tif';
    viablemask = imread(filename1);
    viablemask(viablemask==1) = 255;
    viablemask = uint8(viablemask);
    viablemask = imbinarize(viablemask);
    %figure, imshow(viablemask);
    
    
    filename = baseFilename;
    %disp filename
    svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page); %create an adapter which modulates how the large svs file is accesse
    tic
    %fun =@(block)feature_extraction(block.data);
   
    % Non-overlapping
    blockproc(svs_adapter,tileSize,@fun)     
    toc 
 
    
    % Check if the 70% of teh image area is only background remove that patch,
    % generate the patch and write in disk
    
function fun(block_struct)
    
     global filename
     global filename1
     global viablemask
%     global label_matrix


    a = block_struct.data;
    [row col dim] = size(a);
    
    [viable] = tumor_find(viablemask,block_struct.location(1),block_struct.location(2));
    [row col dim] = size(viable);
    
    
    
    
    
    
    % carefully fix values here
    
    
        J2 = viable;

        %J2 = imresize(viable,.5,'Antialiasing',true); 
        
        %output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/256/Test/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/GTPatches_01_01_0103/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), block_struct.location(1),block_struct.location(2)));
        %imwrite(J1,output_filename);
        imwrite(J2,output_filename1);
        
        
end

