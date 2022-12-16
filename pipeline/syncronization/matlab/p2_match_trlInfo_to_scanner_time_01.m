

%% Correct the trlMat for the 

logTrl  = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\trialInfo_20220513');
logTs   = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded_new_shiftTimeaxis.mat');
scanDat = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513_rec1_00001_i2c_readout.mat');

%% Transform the trl indexes to scanner timeaxes time

shiftLogSec = logTs.shiftLogTimeSec;
trialInfo = logTrl.trialInfo;
trialInfoShiftSec = nan(size(trialInfo));
nEv = size(trialInfo,2);
for iEv = 1:nEv
  if iEv == 2
    trialInfoShiftSec(:,iEv) = trialInfo(:,iEv);
    continue;
  elseif iEv == 10
    noNanInd = ~isnan(trialInfo(:,iEv));
    trialInfoShiftSec(noNanInd,iEv) = trialInfoShiftSec(noNanInd,9) - trialInfoShiftSec(noNanInd,1);
    continue;
  end
  noNanInd = ~isnan(trialInfo(:,iEv));
  trialInfoShiftSec(noNanInd,iEv) = shiftLogSec(trialInfo(noNanInd,iEv));
end

%% Now transform times to scanner frames

frameSec = scanDat.frameTs;
edges = [frameSec; inf];
trialInfoScanFrames = nan(size(trialInfo));
nEv = size(trialInfo,2);
for iEv = 1:nEv
  if iEv == 2
    trialInfoScanFrames(:,iEv) = trialInfo(:,iEv);
    continue;
  elseif iEv == 10
    noNanInd = ~isnan(trialInfo(:,iEv));
    trialInfoScanFrames(noNanInd,iEv) = trialInfoScanFrames(noNanInd,9) - trialInfoScanFrames(noNanInd,1);
    continue;
  end
  noNanInd = ~isnan(trialInfo(:,iEv));
  [~, trialInfoScanFrames(noNanInd,iEv)] = histc(trialInfoShiftSec(noNanInd,iEv),edges);
end

%% Ugly lazy version
cfg = [];
cfg.script = 'p2_match_trlInfo_to_scanner_time_01.m';
saveDir = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData';
saveName = '20220513-165536_677_decoded_new_scan_trlInfo';
save(fullfile(saveDir,saveName),'cfg','trialInfo','trialInfoShiftSec','trialInfoScanFrames')






































































































































