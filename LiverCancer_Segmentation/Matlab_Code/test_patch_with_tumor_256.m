% Take 1024x1024 with overlap 1024x1024 then total size will be 2048x2048


function [] = test_patch_with_tumor_256(name1)

    global filename
    global filename1
    global viablemask
    global test_label_matrix
    global k
    
    test_label_matrix = {};
    k = 1;
    tileSize = [512, 512];

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
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
    
%     global filename
%     global filename1
%     global viablemask
%     global label_matrix


    a = block_struct.data;
    [row col dim] = size(a);
    
    [viable] = tumor_find(viablemask,block_struct.location(1),block_struct.location(2));
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
    
    
    
    
    % carefully fix values here
    
    if ((tissue_region / (512*512) > 0.25) && (count_bg /(512 * 512) < 0.7) && (row == 512) && (col == 512))    %check the ratio of total no of pixels in [512 * 512] image , to remove the background patches.     
        
        %Resize the patch to get patch of size 1024x1024
        J =  block_struct.data;
        patch_name = sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2));
        
        if (count_viable / (512*512) < 0.3)
            
            test_label_matrix{k,1} = patch_name;
            test_label_matrix{k,2} = 0;     % label 0 : non viable tumor region
            k = k+1;
        else
            test_label_matrix{k,1} = patch_name;
            test_label_matrix{k,2} = 1;     % label 1 : viable tumor region
            k = k+ 1;
        end
        
        J1 = imresize(J,.5,'Antialiasing',true); 
        J2 = imresize(viable,.5,'Antialiasing',true); 
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/256/Test/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        output_filename1 = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/256_ViableTumor/Test/',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), block_struct.location(1),block_struct.location(2)));
        imwrite(J1,output_filename);
        imwrite(J2,output_filename1);
        
        
    end
end

fid = fopen( '/raid/data/home/matlab/Mousumi/patch_dataset/256_withGT/256_test.csv', 'a' );
% for jj = 1 : length( train_label_matrix )
%     fprintf( fid, '%s,%d\n', train_label_matrix(jj,1), train_label_matrix(jj,2));
% end

     for k=1:size(test_label_matrix,1)
         fprintf(fid,'%s,%d\n',test_label_matrix{k,:});
     end
 
  fclose(fid);
  
end


