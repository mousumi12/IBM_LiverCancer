

function [J] = region_find(I,y_min, x_min, size)



width = size; 
height = size;


J = imcrop(I,[x_min, y_min, width-1, height-1]);
end