

% Consider patches of size 256x256 at 20x consecutive. Then try to take patches ta
% 10x and 5x centered at same 


function [] = test_patch_with_tumor_256_QualitativeFullWSITest(name1)

    global filename
    global filename1
    global viablemask
    global k
    global test_label_matrix
    global Im
    global newImg
    
    k = 1;
    test_label_matrix = {};
    tileSize = [256, 256];
   

    input_svs_page = 1;   %the page of the svs file we're interested in loading approx 10x
    input_svs_file =  strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
    [~,baseFilename,~]=fileparts(input_svs_file);

    filename1 = strcat('/raid/data/home/matlab/Mousumi/viable_tumors/', strtok(name1,'.svs'),'_viable.tif'); %01_01_0103_viable.tif';
    viablemask = imread(filename1);
    viablemask(viablemask==1) = 255;
    viablemask = uint8(viablemask);
    viablemask = imbinarize(viablemask);
    %figure, imshow(viablemask);
    
    original_image = strcat('/raid/data/home/matlab/Mousumi/Raw_Images/', name1);
    Im = imread(original_image);
    [row, col, dim] = size(Im);
    
    % 
    
    newImg = zeros(row + 384 + 384, col + 384 + 384, 3, 'uint8');   
    newImg(384 : 384 + (row-1), 384 : 384 + (col-1), 1) = Im(:,:,1);
    newImg(384 : 384 + (row-1), 384 : 384 + (col-1), 2) = Im(:,:,2);
    newImg(384 : 384 + (row-1), 384 : 384 + (col-1), 3) = Im(:,:,3);
    
    
    
    filename = baseFilename;
    %disp filename
    svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page); %create an adapter which modulates how the large svs file is accesse
    tic
    %fun =@(block)feature_extraction(block.data);
   
    % Non-overlapping
    blockproc(svs_adapter,tileSize,@fun)    % Don't use overlapping for test patches
    
   
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
    
    if ((tissue_region / (256*256) > 0.25) && (count_bg /(256 * 256) < 0.7) && (row == 256) && (col == 256))    %check the ratio of total no of pixels in [1024 * 1024] image , to remove the background patches.     
        
        
        J =  block_struct.data;
        patch_name = sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2));
        
       
        viable_tumor_Pixels = viable(:,:) ==1 ;
        count_viable = sum(viable_tumor_Pixels(:));
        
        if (count_viable / (256*256) < 0.3)
            
            
            test_label_matrix{k,1} = patch_name;
            test_label_matrix{k,2} = 0;     % label 0 : non viable tumor region
            k= k+1;
        else
            test_label_matrix{k,1} = patch_name;
            test_label_matrix{k,2} = 1;     % label 1 : viable tumor region
            k= k+1;
        end
        
        % ------------------------------------------
        % Now process J to extract patches of size 1024x1024 and 512x512
        % with same center
        
        
        x_cen = row/2; 
        y_cen = col/2;

        % First extract patch of size 512x512
        xmin_L3 = (x_cen - 256);
        ymin_L3 = (y_cen - 256);
        width_L3 = 512;
        
       
        [J_512] = region_find(newImg, xmin_L3, ymin_L3,width_L3);
        
    
        % Extract patch of size 1024x1024
        xmin_L2 = (x_cen - 512);
        ymin_L2 = (y_cen - 512);
        width_L2 = 1024;
        
        [J_1024] = region_find(newImg, xmin_L2, ymin_L2, width_L2);
        
        % ----------------------------------------------------------
        
        
        %%%%%%%%%%%%%%%%%%%%%
        % resize 1024x1024 (J, viable) 2 times to get patches of size
        % 256x256 in 5x resolution
        J1 = imresize(J_1024,.5,'Antialiasing',true); 
        J2_5x = imresize(J1,.5,'Antialiasing',true); 
        
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/5x/img/Test2/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        
        imwrite(J2_5x,output_filename);
        
        % -----------------------------------------------------
        % resize 512x512 (J_512, viable_512) 1 time to get patches of size
        % 256x256 in 10x resolution      
        
        J1_10x = imresize(J_512,.5,'Antialiasing',true); 
        
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/10x/img/Test2/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        
        imwrite(J1_10x,output_filename);
        
        % ------------------------------------------------------
        % save J_256, viable_256 of size 256x256 at 20x resolution 
        
        output_filename = strcat('/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/20x/img/Test2/',sprintf('%s_%d_%d.tiff', filename, block_struct.location(1),block_struct.location(2)));
        
        imwrite(J,output_filename);
        
        
        % FInd the label of the patch based on the label of highest
        % resolution patch at 20x for 256x256
        
        
        
        
    end
end

fid = fopen( '/raid/data/home/matlab/Mousumi/patch_dataset/MultiResolution/256_test2.csv', 'a' );
% for jj = 1 : length( test_label_matrix )
%     fprintf( fid, '%s,%d\n', test_label_matrix(jj,1), test_label_matrix(jj,2));
% end

     for idx=1:size(test_label_matrix,1)
         fprintf(fid,'%s,%d\n',test_label_matrix{idx,:});
     end
 
  fclose(fid);
  
  newImg = [];

end


