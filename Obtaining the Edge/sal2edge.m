data_root = '/GT/';
out_root = '/Output/';
lst_set = '/namelist/';
index_file = fullfile([lst_set '.lst']);

fileID = fopen(index_file);
im_ids = textscan(fileID, '%s');
im_ids = im_ids{1};
fclose(fileID);

num_images = length(im_ids);

for im_id = 1:600

    id = im_ids{im_id};
    id = id(1:end-4);
    
%     img_path = fullfile(data_root, [id '.jpg']);
%     image = imread(img_path);
   
    gt = imread(fullfile(data_root, [id '.png']));
    gt = (gt > 128);
    gt = double(gt);

    [gy, gx] = gradient(gt);
    temp_edge = gy.*gy + gx.*gx;
    temp_edge(temp_edge~=0)=1;
    bound = uint8(temp_edge*255);

    save_path = fullfile(out_root, [id '.png']);
    imwrite(bound, save_path);

end