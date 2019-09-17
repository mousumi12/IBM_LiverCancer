clear all; clc;


I = imread('/raid/data/home/matlab/Mousumi/dataset/Training_phase_1_014/01_01_0101_whole.tif');
I(I==1) = 255;
J = uint8(I);

figure,imshow(J);

% Find image level details

pathname = 'K:\PAIP_Challenge\Raw_Images\';
filename = '01_01_0092.svs';
I1info = imfinfo([pathname filename]);

for i=1:numel(I1info),pageinfo1{i}=['Page ' num2str(i) ': ' num2str(I1info(i).Height) ' x ' num2str(I1info(i).Width)]; end
fprintf('done.\n');
fname=[pathname filename];
if numel(I1info)>1,
    [s,v]=listdlg('Name','Choose Level','PromptString','Select a page for Roi Discovery:','SelectionMode','single','ListSize',[170 120],'ListString',pageinfo1); drawnow;
    if ~v, guidata(hObject, handles); return; end
    fprintf('Reading page %g of image 1... ',s);
    io=imread(fname,s);
    fprintf('done.\n');
else
    fprintf('Image doesnt have any pages!\n');
end