% Take 1024x1024 with overlap 1024x1024 then total size will be 2048x2048

function [] = train_patch_generate_1024(name1)

    global filename
    global filename1
    global I
    
   
    % Read the svs file for highest resolution corresponds to Index = 1
    
    tileSize = [2048, 2048];

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
    
    if ((tissue_region / (2048*2048) > 0.25) && (count_bg / (2048*2048) <0.7) && (row == 2048) && (col == 2048))    %check the ratio of total no of pixels in [512 * 512] image , to remove the background patches.     
        
        %Resize the patch to get patch of size 1024x1024
        J =  block_struct.data;
        
        J1 = imresize(J,.5,'Antialiasing',true);       
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/dataset/1024/train/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        imwrite(J1,output_filename1);
        
%         %Central crop patch of size 1024x1024 from 2048x2048
%         
%         [row, col, dim] = size(J);
%        
%          x_cen = row/2; 
%          y_cen = col/2;
% 
%         
%         xmin_1 = (x_cen - 512);
%         ymin_1 = (y_cen - 512);
%         width_1 = 1023;
%         height_1 = 1023;
%         im1 = imcrop(J,[xmin_1 ymin_1 width_1 height_1]);
%         
%         % Now im1 is 1024x1024 patch at 20x we have to resize to get patch
%         % of size 512x512 at 10x
%         
%         J2 = imresize(im1,.5,'Antialiasing',true);       
%         output_filename2 = strcat('K:\PAIP_Challenge\dataset\512\train\',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
%         imwrite(J2,output_filename2);
%         
%         %Central crop patch of size 512x512 from 2048x2048
%         
%         xmin_2 = (x_cen - 256);
%         ymin_2 = (y_cen - 256);
%         width_2 = 511;
%         height_2 = 511;
%         im2 = imcrop(J,[xmin_2 ymin_2 width_2 height_2]);
%         
%         % Now im2 is 512x512 patch at 20x we have to resize to get patch
%         % of size 256x256 at 10x
%         
%         J3 = imresize(im2,.5,'Antialiasing',true);       
%         output_filename3 = strcat('K:\PAIP_Challenge\dataset\256\train\',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
%         imwrite(J3,output_filename3);
%         
%         %Central crop patch of size 256x256 from 2048x2048
%         
%         xmin_3 = (x_cen - 128);
%         ymin_3 = (y_cen - 128);
%         width_3 = 255;
%         height_3 = 255;
%         im3 = imcrop(J,[xmin_3 ymin_3 width_3 height_3]);
%         
%         % Now im2 is 256x256 patch at 20x we have to resize to get patch
%         % of size 128x128 at 10x
%         
%         J4 = imresize(im3,.5,'Antialiasing',true);       
%         output_filename4 = strcat('K:\PAIP_Challenge\dataset\128\train\',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
%         imwrite(J4,output_filename4);
        
    end
    end


end
