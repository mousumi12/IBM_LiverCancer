clear all; clc;

%global filename 
%global filename1
%global name1
%global I

file1 = dir('/raid/data/home/matlab/Mousumi/Raw_Images/*');
files = file1(3:46);
file = files';

% Split the folders first
% k = 1;
% N = 44;
% x = randperm(N);
% train_set= x(1:32);
% val_set = x(33:39);
% test_set = x(40:44);
% save train_val_test_split_MultiRes.mat train_set val_set test_set

% set1: train_set
load('train_val_test_split_MultiRes_Overlapping.mat')

for i = 1:length(train_set)
    idx = train_set(i);   
    % for each WSI listed in set1 do patch extraction
    ifile = file(idx);
    name1 = ifile.name;
    %train_patch_generate_1024(name1)
    %train_patch_generate_512(name1)
    %train_patch_generate_256(name1)
    %train_patch_generate_128(name1)
   
    %train_patch_generate('01_01_0092.svs')
    %train_patch_with_tumor_256(name1);
    
    train_patch_with_tumor_1024_basePatch(name1)
    
end


%set2: val_set
for i = 1:length(val_set)
    idx = val_set(i);
    
    % for each WSI listed in set1 do patch extraction
    ifile = file(idx);
    name1 = ifile.name;    
    %val_patch_with_tumor_256(name1)
    
    val_patch_with_tumor_1024_basePatch(name1)
end

%set3: test_set


for i = 1:length(test_set)
    idx = test_set(i);
    
    % for each WSI listed in set1 do patch extraction
    ifile = file(idx);
    name1 = ifile.name;
    %test_patch_generate_1024(name1)
    %test_patch_generate_512(name1)
    %test_patch_generate_256(name1)
    %test_patch_generate_128(name1)
    
    %test_patch_with_tumor_256(name1)
   
    %test_patch_with_tumor_1024_basePatch(name1)
    test_patch_with_tumor_256_QualitativeFullWSITest(name1)
end

