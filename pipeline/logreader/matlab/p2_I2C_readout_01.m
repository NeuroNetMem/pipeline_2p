function [i2c, frameTs] = p2_I2C_readout_01(datFilePath)

% addpath(genpath('C:\Users\Jeroen\Documents\Analysis\Code\Projects\Nijmegen\2P'));
% datFilePath = 'C:\Users\Jeroen\Documents\2P\2PData\file_00039.tif';
% datFilePath = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\file_00039.tif';
% addpath(genpath('Downloads'));


tiffInfo = imfinfo(datFilePath);
nFrames = length(tiffInfo);

i2c = struct;
i2c.ts = [];
i2c.val = [];
i2c.frameNum = [];
frameTs = nan(nFrames,1);

for iFrame = 1:nFrames
  i2cStIndx = strfind(tiffInfo(iFrame).ImageDescription,'I2CData = ');
  i2cFrame = tiffInfo(iFrame).ImageDescription(i2cStIndx+11:end);
  
  % First get the timestamps
  i2cRaw = textscan(i2cFrame,'%s');
  i2cRaw = i2cRaw{1};
  i2cTsTemp = i2cRaw(1:2:end-1);
  
  if ~isempty(i2cTsTemp)

      nTrigs = size(i2cTsTemp,1);
      tempTs = [i2cTsTemp{:}];

      stInd = strfind(tempTs,'{');
      ndInd = strfind(tempTs,',');
      i2cTs = nan(nTrigs,1);
      for iTrig = 1:nTrigs
        i2cTs(iTrig) = str2num(tempTs(stInd(iTrig)+1:ndInd(iTrig)-1));
      end

      i2c.ts = [i2c.ts;i2cTs];

      % Now get the values
      i2cValsTemp = i2cRaw(2:2:end-1);
      tempVals = [i2cValsTemp{:}];
      stInd = strfind(tempVals,'[');
      ndInd = strfind(tempVals,']');
      i2cVals = nan(nTrigs,1);
      for iTrig = 1:nTrigs
          i2cVal = str2num(tempVals(stInd(iTrig)+1:ndInd(iTrig)-1));
          if length(i2cVal) > 1
            disp(['We have a packet with more than 1 value in i2c event ' num2str(iFrame) ' we used the first of these values: ' (num2str(i2cVal))])
            i2cVals(iTrig) = i2cVal(1);
          else
            i2cVals(iTrig) = i2cVal;
          end
      end

      i2c.val = [i2c.val;i2cVals];

      % Add a frame number index just to be sure
      i2c.frameNum = [i2c.frameNum; ones(nTrigs,1)*iFrame];
      
  end
  
  % Get timestamps
  tsStIndx = strfind(tiffInfo(iFrame).ImageDescription,'frameTimestamps_sec = ');
  tsNdIndx = strfind(tiffInfo(iFrame).ImageDescription,'acqTriggerTimestamps_sec = ');
  frameTs(iFrame) = str2num(tiffInfo(iFrame).ImageDescription(tsStIndx+22:tsNdIndx-2));
  
end


% % datFile = '/Users/bos/Downloads/file_00039.tif';
% datFile = 'C:\Users\Jeroen\Documents\2P\2PData\file_00039.tif';
% 
% %% How do we do this?
% 
% tiffInfo = imfinfo(datFile);
% tiffInfo(1).ImageDescription
% tiffInfo(4).ImageDescription
% % i2cStIndx = strfind(temp4,'I2CData = ');
% 
% iFrame = 1;
% i2cStIndx = strfind(tiffInfo(iFrame).ImageDescription,'I2CData = ');
% i2cFrame = tiffInfo(iFrame).ImageDescription(i2cStIndx+11:end);
% 
% % i2Ev = textscan(i2cFrame,'%s','Delimiter','{');
% % Z = textscan(str,'%s','Delimiter',' ')';
% % Z{:}'
% 
% 
% i2cRaw = textscan(i2cFrame,'%s');
% i2cRaw = i2cRaw{1};
% i2cTsTemp = i2cRaw(1:2:end-1);
% i2cValsTemp = i2cRaw(2:2:end-1);
% 
% nTrigs = size(i2cTsTemp,1);
% tempTs = [i2cTsTemp{:}];
% 
% tempTs(strfind(tempTs,'{')) = [];
% tempTs(strfind(tempTs,',')) = [];
% 
% negIndx = strfind(tempTs,'-');
% negInd = ((negIndx-1)/12)+1;
% tempTs(negIndx) = [];
% 
% i2cTs = str2num(reshape(tempTs,11,nTrigs)');
% i2cTs(negInd) = -i2cTs(negInd);
% 
% tempVals = [i2cValsTemp{:}];
% tempVals(strfind(tempVals,'}')) = [];
% tempVals(strfind(tempVals,'[')) = [];
% tempVals(strfind(tempVals,']')) = [];
% i2cVals = str2num(reshape(tempVals,1,nTrigs)'); % This only works with values from 0 to 9
% 
% %% Unfortunately this is too rigid. If the number of elements changes we get into trouble
% 
% 
% stInd = strfind(tempTs,'{');
% ndInd = strfind(tempTs,',');
% nEntries = length(stInd);
% i2cTs = nan(nEntries,1);
% for iTrig = 1:nEntries
%   i2cTs(iTrig) = str2num(tempTs(stInd(iTrig)+1:ndInd(iTrig)-1));
% end
% 
% %% Now for the real deal
% 
% datFile = 'C:\Users\Jeroen\Documents\2P\2PData\file_00039.tif';
% 
% tiffInfo = imfinfo(datFile);
% nFrames = length(tiffInfo);
% 
% i2c = struct;
% i2c.ts = [];
% i2c.val = [];
% i2c.frameNum = [];
% 
% for iFrame = 1:nFrames
%   i2cStIndx = strfind(tiffInfo(iFrame).ImageDescription,'I2CData = ');
%   i2cFrame = tiffInfo(iFrame).ImageDescription(i2cStIndx+11:end);
%   
%   % First get the timestamps
%   i2cRaw = textscan(i2cFrame,'%s');
%   i2cRaw = i2cRaw{1};
%   i2cTsTemp = i2cRaw(1:2:end-1);
% 
%   nTrigs = size(i2cTsTemp,1);
%   tempTs = [i2cTsTemp{:}];
% 
%   tempTs(strfind(tempTs,'{')) = [];
%   tempTs(strfind(tempTs,',')) = [];
% 
%   negIndx = strfind(tempTs,'-');
%   negInd = ((negIndx-1)/12)+1;
%   tempTs(negIndx) = [];
% 
%   i2cTs = str2num(reshape(tempTs,11,nTrigs)');
%   i2cTs(negInd) = -i2cTs(negInd);
%   i2c.ts = [i2c.ts;i2cTs];
% 
%   % Now get the values
%   i2cValsTemp = i2cRaw(2:2:end-1);
%   tempVals = [i2cValsTemp{:}];
%   tempVals(strfind(tempVals,'}')) = [];
%   tempVals(strfind(tempVals,'[')) = [];
%   tempVals(strfind(tempVals,']')) = [];
%   i2cVals = str2num(reshape(tempVals,1,nTrigs)'); % This only works with values from 0 to 9
%   i2c.val = [i2c.val;i2cVals];
%   
%   % Add a frame number index just to be sure
%   i2c.frameNum = [i2c.frameNum; ones(nTrigs,1)*iFrame];
% end
% 
% %% Now for a working version
% 
% datFile = 'C:\Users\Jeroen\Documents\2P\2PData\file_00039.tif';
% 
% tiffInfo = imfinfo(datFile);
% nFrames = length(tiffInfo);
% 
% i2c = struct;
% i2c.ts = [];
% i2c.val = [];
% i2c.frameNum = [];
% 
% for iFrame = 1:nFrames
%   i2cStIndx = strfind(tiffInfo(iFrame).ImageDescription,'I2CData = ');
%   i2cFrame = tiffInfo(iFrame).ImageDescription(i2cStIndx+11:end);
%   
%   % First get the timestamps
%   i2cRaw = textscan(i2cFrame,'%s');
%   i2cRaw = i2cRaw{1};
%   i2cTsTemp = i2cRaw(1:2:end-1);
% 
%   nTrigs = size(i2cTsTemp,1);
%   tempTs = [i2cTsTemp{:}];
% 
%   stInd = strfind(tempTs,'{');
%   ndInd = strfind(tempTs,',');
%   i2cTs = nan(nTrigs,1);
%   for iTrig = 1:nTrigs
%     i2cTs(iTrig) = str2num(tempTs(stInd(iTrig)+1:ndInd(iTrig)-1));
%   end
% 
%   i2c.ts = [i2c.ts;i2cTs];
% 
%   % Now get the values
%   i2cValsTemp = i2cRaw(2:2:end-1);
%   tempVals = [i2cValsTemp{:}];
%   stInd = strfind(tempVals,'[');
%   ndInd = strfind(tempVals,']');
%   i2cVals = nan(nTrigs,1);
%   for iTrig = 1:nTrigs
%     i2cVals(iTrig) = str2num(tempVals(stInd(iTrig)+1:ndInd(iTrig)-1));
%   end
%   
%   i2c.val = [i2c.val;i2cVals];
% 
%   % Add a frame number index just to be sure
%   i2c.frameNum = [i2c.frameNum; ones(nTrigs,1)*iFrame];
%   
% end
















































































