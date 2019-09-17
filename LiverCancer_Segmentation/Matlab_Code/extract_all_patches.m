clear all; clc;

global filename
    global filename1
    global viablemask
    global test_label_matrix
    global k
    

    name1 = '01_01_0100.svs';
    k = 1;
    tileSize = [512, 512];

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
    [~,baseFilename,~]=fileparts(input_svs_file);

    filename1 = strcat('/raid/data/home/matlab/Mousumi/viable_tumors/', strtok(name1,'.svs'),'_viable.tif'); %01_01_0103_viable.tif';
    viablemask = imread(filename1);
    viablemask(viablemask==1) = 255;
    viablemask = uint8(viablemask);
    %viablemask = imbinarize(viablemask);
    %figure, imshow(viablemask);
    
    
    filename = baseFilename;
    %disp filename
    svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page); %create an adapter which modulates how the large svs file is accesse
    tic
    %fun =@(block)feature_extraction(block.data);
   
    % Non-overlapping
    blockproc(svs_adapter,tileSize,@fun)     
    toc 
 
    

    
function fun(block_struct)
    
    global filename
    global filename1
    global viablemask
    global label_matrix


    a = block_struct.data;
    [row col dim] = size(a);
    
    [viable] = tumor_find(viablemask,block_struct.location(1),block_struct.location(2));
    [row col dim] = size(viable);
    
 
    
    % carefully fix values here
    
     
        
        %Resize the patch to get patch of size 1024x1024
        J =  block_struct.data;
        patch_name = sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2));
        
        
        J1 = imresize(J,.5,'Antialiasing',true); 
        %J2 = imresize(viable,.5,'Antialiasing',true); 
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/ForSizeEstimate/01_01_0100/Image/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        %output_filename1 = strcat('/raid/data/home/matlab/Mousumi/Test_01_01_0113/GT/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), block_struct.location(1),block_struct.location(2)));
        imwrite(J1,output_filename);
        %imwrite(J2,output_filename1);
        
        
    
end

