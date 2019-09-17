% Take 1024x1024 with overlap 1024x1024 then total size will be 2048x2048

function [] = test_patch_generate_128(name1)

    global filename
    global filename1
    global I
    
   
    % Read the svs file for highest resolution corresponds to Index = 1
    
    tileSize = [256, 256];

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
    [~,baseFilename,~]=fileparts(input_svs_file);

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

    a = block_struct.data;
    [row col dim] = size(a);
        
    %figure,imshow(uint8(a));
    a_gray = rgb2gray(a);
    a_bin = imbinarize(a_gray);
    %figure,imshow(a_gray);
    a_bin = logical(a_bin);
    %figure,imshow(a_bin);
    tissue_region = sum(a_bin(:)==0); % this gives the number of pixels inside the tissue region
    tissue_bg  = a(:,:,1) >= 210 & a(:,:,2) >= 210 & a(:,:,3) >=210;
    count_bg = sum(tissue_bg(:));
    
    % carefully fix values here
    
    if ((tissue_region / (256*256) > 0.25) && (count_bg / (256*256) <0.7) && (row == 256) && (col == 256))    %check the ratio of total no of pixels in [512 * 512] image , to remove the background patches.     
        
        %Resize the patch to get patch of size 1024x1024
        J =  block_struct.data;
        
        J1 = imresize(J,.5,'Antialiasing',true);       
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/dataset/128/test/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        imwrite(J1,output_filename1);
        
        
        
    end
    end


end
