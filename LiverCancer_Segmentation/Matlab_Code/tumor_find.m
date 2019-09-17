

function [J] = tumor_find(I,y_min, x_min)

width =  256;   %1024; 
height = 256 ;   %1024;
J = imcrop(I,[x_min, y_min, width-1, height-1]);
end