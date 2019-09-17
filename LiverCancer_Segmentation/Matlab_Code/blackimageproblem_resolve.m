clear all; clc;

%input_svs_page=3; %the page of the svs file we're interested in loading
%input_svs_file = strcat('K:\PAIP_Challenge\Raw_Images\', '01_01_0092.svs');


files = dir('K:\PAIP_Challenge\Raw_Images\*.svs');
file = files';

for i = 1:length(file)
I = imread(strcat('K:\PAIP_Challenge\Raw_Images\',file(i).name)); %01_01_0136.svs
%figure,imshow(I,[]);
% I_R = I(:,:,1);
% I_G = I(:,:,2);
% I_B = I(:,:,3);

if (I == 0)
    f = msgbox('Black');
    
end
end

% tileSize = [2048, 2048];
%     [~,baseFilename,~]=fileparts(input_svs_file);
% 
%     filename = baseFilename;
%     %disp filename
%     svs_adapter = PagedTiffAdapter(input_svs_file,input_svs_page); %create an adapter which modulates how the large svs file is accesse
%     tic
%     %fun =@(block)feature_extraction(block.data);
%     fun=@(block) imwrite(block.data,strcat('K:\PAIP_Challenge\dataset\1024\train\',sprintf('%s_%d_%d.png',baseFilename,block.location(1),block.location(2)))); %make a function which saves the individual tile with the row/column information in the filename so that we can refind this tile later
%    blockproc(svs_adapter,tileSize,fun); %perform the splitting
%    
   
