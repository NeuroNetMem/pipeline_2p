

%% Scanner side

%load('C:\Users\Jeroen\Downloads\i2cDataB1B2.mat')

% scanImageRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\';
% scanImageFilename = 'i2cDataB1B2.mat';
% scanImageRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\audioseq_20220314\';
% scanImageFilename = 'i2cReadout.mat';
scanImageRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\';
scanImageFilename = '20220513_rec1_00001_i2c_readout.mat';
scanDat = load(fullfile(scanImageRoot,scanImageFilename));

scOnsetInd = scanDat.i2c.val == 1;
scOnsetSec = scanDat.i2c.ts(scOnsetInd);
dScSec = diff(scOnsetSec); % Time is in seconds

%% Behavioral side
% Load data

% load('C:\Users\Jeroen\Documents\Nijmegen\VR\VR_dataPreprocessed\20220119-165038_863_decodedFile.mat');
% logRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\';
% logFilename = '20220120-180442_339_decoded.mat';
% logRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\audioseq_20220314\';
% logFilename = '20220314-210122_358_decoded.mat';
logRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\';
logFilename = '20220513-165536_677_decoded_new.mat';
logDat = load(fullfile(logRoot,logFilename));

%% Get Digital Inputs
 
digitalIn = double(logDat.digitalIn);

nChan = size(digitalIn,2);
for iChan = 1:nChan
  dChan = diff(digitalIn(:,iChan));
  digIn.onset{iChan} = find(dChan == 1) + 1;
  digIn.offset{iChan} = find(dChan == -1);
  % we should add a boundry check to not have to assume pairs
end
% what is in where?
% which have stuff: 1vid1 2vid2 3iti 8? 9scanClock 10? 13belt? 14belt? 
% changed it to ITI, vid 1, vid2 etc to vid6
digIn.labels = {'iti','sound1','sound2','sound3', ...
  'sound4','sound5','sound6','empty', ...
  'frameclock','empty','empty','empty', ...
  'A_wheel','B_wheel','empty','empty'};
% digIn.labels = {'vid1','vid2','iti','empty', ...
%   'empty','empty','empty','empty', ...
%   'frameclock','empty','empty','empty', ...
%   'A_wheel','B_wheel','empty','empty'};

%% Get Digital Outputs

digitalOut = double(logDat.digitalOut);

nChan = size(digitalOut,2);
for iChan = 1:nChan
  dChan = diff(digitalOut(:,iChan));
  digOut.onset{iChan} = find(dChan == 1) + 1;
  digOut.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% which channels have stuff 4sync?
digOut.labels = {'empty','empty','empty','sync', ...
  'empty','empty','empty','empty'};

%% logDat is in packets, so we have to change to time (seconds) USED STARTS OF THE LOGFILE
% logDat.startTs should be in uSeconds

syncChan = 4; % Inverted should be channel 5
logStartTs = double(logDat.startTS);
logOnsetSec = logStartTs(digOut.onset{syncChan})/10^6;
dLogSec = diff(logOnsetSec)';

%% Get the shift value
% Scanner
%   scOnsetSec
%   dScSec
% Logfile
%   logOnsetSec
%   dLogSec

% dNpEvOnset = diff(npEv.syncSamplesLFP(npEv.onsetLFP))/2.5;
% dLogEvOnset = diff(digOut.onset{4});
% temp = dLogSec;
% dLogSec = dScSec;
% dScSec = temp;

figure; hold on
plot(dLogSec)
plot(dScSec)
legend('LogFile','Scanner')

%%%%%%%%%%%%%%%%%%%%%%
% Match with the shortest vars
nSc = length(dScSec);
nLog = length(dLogSec);

if nLog > nSc
  longVec = dLogSec;
  shortVec = dScSec;
  orderVecs = 1;
else
  longVec = dScSec;
  shortVec = dLogSec;
  orderVecs = 0;
end

vecDiff = length(longVec) - length(shortVec);
% Do we assume the smaller vec is always 100% within the bigger one?
% This assumption is wrong! Even though it should be if we do things in
% order
nOffsets = 2*vecDiff; % Enough?
padd = mean(longVec);
paddLongVec = [longVec; ones(vecDiff,1)*padd];
offsetCorr = nan(nOffsets,1);
for iOffset = 1:nOffsets
  subLongVec = paddLongVec(iOffset:length(shortVec)+iOffset-1);
  offsetCorr(iOffset) = corr(shortVec,subLongVec,'rows','complete');
end

[maxCorr, mInd] = max(offsetCorr);
shift = mInd;
disp(['% Correlation value = ' num2str(maxCorr)])
if maxCorr < 0.95
  disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
  disp(['% LOW CORRELATION VALUE ' num2str(maxCorr)])
  disp('%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%')
end

plotShortVec = [nan(shift-1,1); shortVec];
figure; hold on
plot(plotShortVec)
plot(longVec)
if orderVecs == 1
  legend('Scanner','LogFile')
  title(['Shift = ' num2str(shift) ' Logfile barcode ' num2str(shift) ' corresponds to scanner 1'])
else
  legend('LogFile','Scanner')
  title(['Shift = ' num2str(shift) ' Logfile barcode 1 corresponds to scanner ' num2str(shift)])
end

% Adjust the timeaxis
if orderVecs == 1
  shiftLogTimeSec = double(logDat.startTS)/10^6 - (logOnsetSec(shift) - scOnsetSec(1));
else
  shiftLogTimeSec = double(logDat.startTS)/10^6 + (scOnsetSec(shift) - logOnsetSec(1));
end


%% Ugly lazy version
cfg = [];
cfg.script = 'p2_synch_scan_log_01.m';
saveDir = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData';
saveName = '20220513-165536_677_decoded_new_shiftTimeaxis';
save(fullfile(saveDir,saveName),'cfg','shiftLogTimeSec')

%% To do

% fishish shifted log ts
% % convert the trlMat into shifted times
% make trlMat into scanner frame num

% ScanClock 0s        0.033     0.066
% frameNum  1         2         3
% TS        |         |         |
% event1             |
% event2               |
% event3       |


%% Check consistency for Morgane

logVec = plotShortVec(1:end-2);
scanVec = longVec;

offsetVec = scanVec - logVec;
[logVec(end-20:end) scanVec(end-20:end) offsetVec(end-20:end)]
nanmean(offsetVec)



%% Old stuff



% 
% 
% scVec = nan(lengthVec,1);
% scVec(shift:nSc+shift-1) = dScSec;
% figure; hold on
% plot(dLogSec)
% plot(scVec)
% legend('LogFile','Scanner')
% title(['Shift = ' num2str(shift)])
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% 
% % Symmetry %%% Does it need symmetry?
% nSc = length(dScSec);
% nLog = length(dLogSec);
% %maxOffset = abs(nSc - nLog)+1; % just double it. 1 should be enough as well
% maxOffset = max(nSc,nLog);
% symmOffset = [-maxOffset:1:-1 1:1:maxOffset]; % SHould we add the 0?
% nOffsets = length(symmOffset);
% offsetCorr = nan(nOffsets,1);
% logVec = [nan(maxOffset,1); dLogSec; nan(1,1)];
% lengthVec = 2*length(logVec);
% 
% logVec2 = nan(lengthVec,1);
% logVec2(1:length(logVec),1) = logVec;
% for iOffset = 1:nOffsets
% %   iOffset = symmOffset(iOffset);
%   scVec = nan(lengthVec,1);
%   scVec(iOffset:nSc+iOffset-1) = dScSec;
%   offsetCorr(iOffset) = corr(logVec2,scVec,'rows','complete');
% end
% 
% 
% 
% scVec = nan(lengthVec,1);
% scVec(shift:nSc+shift-1) = dScSec;
% figure; hold on
% plot(dLogSec)
% plot(scVec)
% legend('LogFile','Scanner')
% title(['Shift = ' num2str(shift)])
% 
% % This bit get different if the shift is negative, but should it ever be?
% 
% % logOnsetSec = logStartTs(digOut.onset{syncChan})/10^6;
% % scOnsetSec = scanDat.i2c.ts(scOnsetInd);
% 
% display(['shiftValue = ' num2str(shift)])
% display(['This means that ' num2str(logOnsetSec(shift)) ' in the log and ' num2str(scOnsetSec(1)) ' in the scanner happen at the same time'])
% 
% 
% 
% 
% 
% 
% 
% 






















































