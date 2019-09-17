clear all; clc;

%I = imread('/raid/data/home/matlab/Mousumi/viable_tumors/01_01_0103_viable.tif');
%[row col dim] = size(I);
input_svs_file = '/raid/data/home/matlab/Mousumi/Raw_Images/01_01_0113.svs';
input_svs_page = 1;	
tileSize = [512, 512];
svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page);
[~,baseFilename,~]=fileparts(input_svs_file);
filename = baseFilename;

    tic
    outFile= '/raid/data/home/matlab/Mousumi/Result_WholeSlide_Images/01_01_0113_UNet.tif'; %desired output filename
    inFileInfo=imfinfo(input_svs_file); %need to figure out what the final output size should be to create the emtpy tif that will be filled in
    inFileInfo=inFileInfo(input_svs_page); %imfinfo returns a struct for each individual page, we again select the page we're interested in
     
    outFileWriter = bigTiffWriter(outFile, inFileInfo.Height, inFileInfo.Width, tileSize(1), tileSize(1), false); %true   %create another image adapter for output writing
     
    fun=@(block) repmat(imread(strcat('/raid/data/home/matlab/Mousumi/Prediction_Images_UNet/01_01_0113_resized/',sprintf('%s_%d_%d.tiff',filename,block.location(1),block.location(2)))),[1 1 3]); %,1.5); %load the output image, which has an expected filename (the two locations added). In this case my output is 60% smaller than the original image, so i'll scale it back up
     
    %fun=@(block) repmat(imread(strcat(' ',sprintf('%s_%d_%d.tiff', strcat(filename,'_viable'), block.location(1), block.location(2)))),[1 1 3]); %,1.5); 
    
    blockproc(svs_adapter,tileSize,fun,'Destination',outFileWriter); %do the blockproc again, which will result in the same row/column coordinates, except now we specify the output image adatper to write the flie outwards
     
    outFileWriter.close(); %close the file when we're done
    toc

