clear all; clc;

files = dir('../Raw_Images/*.svs');
file = files';
lst = {};

for i = 32:length(file)
    ifile = file(i);
    name1 = ifile.name;
    I = imread(strcat('/raid/data/home/matlab/Mousumi/Raw_Images/',name1));
    if I(:,:,1) == 0
        lst{end+1} = name1 ;  
    end
end