function write32bittiff(data,filename)

    [pathstr,file,ext] = fileparts(filename);
    
    % 32 Bit:
    t = Tiff(filename,'w'); % way to write 32bit-Tiff-files
    
    tagstruct.ImageLength     = size(data,1);
    tagstruct.ImageWidth      = size(data,2);
    tagstruct.Photometric     = Tiff.Photometric.MinIsBlack;
    tagstruct.BitsPerSample   = 32;
    tagstruct.SamplesPerPixel = 1;
    tagstruct.RowsPerStrip    = 16;
    tagstruct.PlanarConfiguration = Tiff.PlanarConfiguration.Chunky;
    tagstruct.Software        = 'MATLAB';

    t.setTag(tagstruct)
    t.write(uint32(data));
    t.close();
    
    disp(strcat('wrote ',file,' in ', pathstr,'!'));
    %-----------------------------------------------------
end
