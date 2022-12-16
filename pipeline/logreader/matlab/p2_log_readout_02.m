
% packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded.mat');
% origPacket = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded.mat');

% packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded_new.mat');

packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\Croc\20221122\20221122-170312_853_decoded.mat');

packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20221201-102828_264_decoded.mat');


%% In this session there are no events for environment C
% The tunnel events are not 100% relyable

%% Digital Inputs

% % % Load file
% tempInput = double(packet.digitalIn);
% % sum(digitalIn)
% goodOrder=[8:-1:1 16:-1:9];
% nChan = size(tempInput,2);
% digitalIn = nan(size(tempInput));
% for iCh = 1:nChan
%     iChan = goodOrder(iCh);
%     digitalIn(:,iChan) = tempInput(:,iCh);
% end

%%%% THESE ARE INVERTED 1 = 16 and 16 = 1

digitalIn = double(packet.digitalIn);
digitalIn = fliplr(digitalIn);

nChan = size(digitalIn,2);
for iChan = 1:nChan
  dChan = diff(digitalIn(:,iChan));
  digIn.onset{iChan} = find(dChan == 1) + 1;
  digIn.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% what is in where?

% digIn.labels = {'empty','empty','wheelA','wheelB', ...
%   'wheelC','IRcamera','sound','scanner', ...
%   'reward_zone','environment1','environment2','environment3', ...
%   'tunnel1','tunnel2','empty','empty'};


% which have stuff: 1vid1 2vid2 3iti 8? 9scanClock 10? 13belt? 14belt? 
digIn.labels = {'empty','empty','wheelA','wheelB', ...
  'wheelC','IRcamera','scanner','sound', ...
  '??','reward_zone','environment1','environment2', ...
  'environment3_broken','tunnel1','tunnel2','environment3'};


%% Plot some stuff 01-12

figure; hold on
plot(digitalIn(:,11)==0)
plot(digitalIn(:,12)*2)
plot(digitalIn(:,16)*3)
plot(digitalIn(:,8)*4)
plot(digitalIn(:,10)*5)
plot((digitalIn(:,14)==0)*6)
plot((digitalIn(:,15)==0)*7)
% plot(digitalOut(:,1)*8)

%%
figure; hold on
plot(digitalIn(:,11))
plot(digitalIn(:,12)*2)
plot(digitalIn(:,16)*3)
plot(digitalIn(:,8)*4)
plot(digitalIn(:,10)*5)
plot((digitalIn(:,14))*6)
plot((digitalIn(:,15))*7)

%% We still have the tunnel switches, 1-12 seems there is one around 1.175 ish
% first invert channels down
tmpOn = digIn.onset{14};
tmpOff = digIn.offset{14};
onset = tmpOn(1:11);
offset = tmpOn(13:end);
onset = [onset; tmpOff(13:end)];
offset = sort([offset; tmpOff(1:12)]);

digIn.onset{14} = onset;
digIn.offset{14} = offset;

plot(digIn.onset{14},ones(length(digIn.onset{14}),1)*6,'dk')
plot(digIn.offset{14},ones(length(digIn.offset{14}),1)*6,'db')

%% tunnel 2
tmpOn = digIn.onset{15};
tmpOff = digIn.offset{15};
onset = tmpOn(1:11);
offset = tmpOn(13:end);
onset = [onset; tmpOff(13:end)];
offset = sort([offset; tmpOff(1:12)]);

digIn.onset{15} = onset;
digIn.offset{15} = offset;

plot(digIn.onset{15},ones(length(digIn.onset{15}),1)*7,'dr')
plot(digIn.offset{15},ones(length(digIn.offset{15}),1)*7,'dm')

%%

saveRoot = 'C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\';
day = '20221201';
filename = [day '_digIn'];
save(fullfile(saveRoot,filename),'digIn')

%% Can we recreate environment tresholds? And clean up the tunnels

digitalInput = double(digitalIn);

envA = digitalInput(:,6) == 0;

% Not all trials have sounds, so we cannot always use them

allTunnel1 = [digIn.onset{3}; digIn.offset{3}];
allTunnel2 = [digIn.onset{2}; digIn.offset{2}];

allTunnel1([1 7 25 28]) = [];
allTunnel2([1 7 25]) = [];

tun1Y = ones(length(allTunnel1),1);
tun2Y = ones(length(allTunnel2),1);

figure; hold on
plot(envA)
plot(digitalInput(:,5)*2)
plot(digitalInput(:,7)*2.2)
% reward zone
plot([allTunnel1 allTunnel1]',[tun1Y*0 tun1Y*2.4]','k');
plot([allTunnel2 allTunnel2]',[tun2Y*0 tun2Y*2.8]','k');

% legend('envA','envB','rew','tun1','tun2')

%% Now that it is clean the envs should start 1500 unities before the start of each tunnel1

sAllTunnel1 = sort(allTunnel1);
allTunnel1St = sAllTunnel1(1:2:end);
tun1StY = ones(length(allTunnel1St),1);

plot([allTunnel1St allTunnel1St]', [tun1StY*0 tun1StY*2.45]','y--')

allTunnel1Nd = sAllTunnel1(2:2:end);
tun1NdY = ones(length(allTunnel1Nd),1);
plot([allTunnel1Nd allTunnel1Nd]', [tun1NdY*0 tun1NdY*2.45]','r--')

tunnel1.onset = allTunnel1St;
tunnel1.offset = allTunnel1Nd;

sAllTunnel2 = sort(allTunnel2);
sAllTunnel2(14) = [];
allTunnel2St = sAllTunnel2(1:2:end);
tun2StY = ones(length(allTunnel2St),1);

plot([allTunnel2St allTunnel2St]', [tun2StY*0 tun2StY*2.85]','m--')

allTunnel2Nd = sAllTunnel2(2:2:end);
tun2NdY = ones(length(allTunnel2Nd),1);
plot([allTunnel2Nd allTunnel2Nd]', [tun2NdY*0 tun2NdY*2.85]','c--')

% Take the distances
vrDist = double(packet.longVar(:,2));
tunnel1Dist = vrDist(allTunnel1St);

tunnel2.onset = allTunnel2St;
tunnel2.offset = [allTunnel2Nd; nan];

% What are the distance onsets of the other environments

envAOn = digIn.offset{6};
envBOn = digIn.onset{5};
envAOn(3) = [];

envAY = ones(length(envAOn),1);
envBY = ones(length(envBOn),1);

plot([envAOn envAOn]', [envAY*0 envAY*1.05]','m--')
plot([envBOn envBOn]', [envBY*0 envBY*2.05]','c--')

% Take all the yellow times, get the corresponding distances
% deduct 1500 and see which match with the ones we have and which ones are
% new. The new ones should be envC

tunnel1Dist = vrDist(allTunnel1St);
envDist = tunnel1Dist - 1500;

addpath(genpath('C:\Users\Jeroen\Documents\Analysis\Code\Projects\Nijmegen\General'))

[N,envIndx] = nearest_array(vrDist,envDist); % When tied the first one is taken
plot(envIndx,ones(length(envIndx),1)*0.5,'r*')

% there is one very large gap between trials... what happened there?

%%

% WHY IS THE LAST TRIAL NOT MARKED AS A C

% Add a random value to envIndx to check which environment we should be in
tmpEvntIndx = envIndx+1000;

inEnvInd = zeros(1,length(envIndx));
eInd = envA(tmpEvntIndx);
inEnvInd(eInd) = 1;
eInd = logical(digitalInput(tmpEvntIndx,5));
inEnvInd(eInd) = 2;

envCInd = inEnvInd == 0;

plot(envIndx(envCInd),ones(sum(envCInd),1)*0.5,'g*')

envCTmp = envIndx(envCInd);

% sAllTunnel2 = sort(allTunnel2);
[~,envTunIndx] = nearest_array(allTunnel2,envCTmp);

envCIndx = allTunnel2(envTunIndx); % onset
plot(envCIndx,ones(length(envCIndx),1)*0.5,'b*')

% which are the corresponding offsets

edges = [1; allTunnel1St];
[a,b] = histc(envCIndx,edges);
plot(edges(b+1),ones(length(b),1)*0.5,'c*')

envC.onset = envCIndx;
envC.offset = edges(b+1);

clear envA
envA.onset = envAOn;
envA.offset = digIn.onset{6};
envA.offset([1 4]) = [];

envB.onset = envBOn;
envB.offset = digIn.offset{5};

allEnv = sort([envA.onset; envB.onset; envC.onset]);

%% When were the sounds and the valves

plot(digIn.onset{9},ones(length(digIn.onset{9}),1)*1.7,'cd')
plot(digIn.offset{9},ones(length(digIn.offset{9}),1)*1.7,'rd')

sound.onset = digIn.onset{9};
sound.offset = digIn.offset{9};

% valve is digOut{8}

plot(digOut.onset{8},ones(length(digOut.onset{8}),1)*2.7,'')


%% Plot stuff together

figure; hold on
for iChan = [2 3 4 6 7 13 16]%1:nChan
  plot(digitalIn(:,iChan)*iChan)
end

legend('sound','rewardZone','env1','env2','env3','tunnel1','tunnel2')

%% Plot stuff separate

for iChan = 1:nChan
  figure
  plot(digitalIn(:,iChan)*iChan)
  title(['Chan ' num2str(iChan)])
end

%% Whats what
% channel map

% 1 'empty'
% 2 'empty'
% 3 'wheelA'
% 4 'wheelB'
% 5 'wheelC'
% 6 'IRcamera'
% 7 'sound'
% 8 'scanner'
% 9 'reward_zone'
% 10 'environment1'
% 11 'environment2'
% 12 'environment3'
% 13 'tunnel1'
% 14 'tunnel2'
% 15 'empty'
% 16 'empty'

% OLD PRE SWITCHING
% digdital input. what we see in matlab
% 1 IR camera ??
% 2 Sound
% 3 Something IR camera??
% 4 Encoder
% 5 Wheel
% 6 Wheel
% 7 broken
% 8 broken
% 9 broken
% 10 broken
% 11 tunnel 2
% 12 tunnel 1
% 13 env 3
% 14 env 2
% 15 env 1          inverted
% 16 reward area


% Missing sound trigger in env 1?
% The distances do not really make sense check


%% Digital Outputs
% Load file
% tempOutput = double(packet.digitalOut);
% % sum(digitalIn)
% goodOrder= 8:-1:1;
% nChan = size(tempOutput,2);
% digitalOut = nan(size(tempOutput));
% for iCh = 1:nChan
%     iChan = goodOrder(iCh);
%     digitalOut(:,iChan) = tempOutput(:,iCh);
% end
% 
% % Load file
% % digitalOut = double(packet.digitalOut);
% % sum(digitalIn)

digitalOut = double(packet.digitalOut);
digitalOut = fliplr(digitalOut);

% digitalOut = double(packet.digitalOut);
nChan = size(digitalOut,2);
for iChan = 1:nChan
  dChan = diff(digitalOut(:,iChan));
  digOut.onset{iChan} = find(dChan == 1) + 1;
  digOut.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% which channels have stuff 4sync?
digOut.labels = {'valve','empty','IR_LED_sync','empty', ...
  'barcode','IR_LED','lick','empty'};

%% Plot stuff

figure; hold on
for iChan = [1:2 4 6 8] %[1:3 5:8]
  plot(digitalOut(:,iChan)*iChan)
end


% 1 empty
% 2 = lick
% 3 epmty
% 4 = barcode
% 5 empty
% 6 empty
% 7 = valve ?
% 8 broken

%% Find inverted channels
% This fails on the short segment of data, but should still be ok for
% larger sessions. We Can look for a way to improve

nChan = size(digitalIn,2);
invChanInd = false(1,nChan);
for iChan = 1:nChan
  nOn = length(digIn.onset{iChan});
  nOff = length(digIn.offset{iChan});
  nEv = min([nOn nOff]);
  if nEv == 0; continue; end
  medFor = median(digIn.offset{iChan}(1:nEv) - digIn.onset{iChan}(1:nEv));
  medInf = median(digIn.onset{iChan}(2:nEv) - digIn.offset{iChan}(1:nEv-1));
  if medInf < medFor
    invChanInd(iChan) = true;
  end
end

% QUICK FIX FOR NOW
%invChanInd(11) = false;

invChanInd([4 7]) = false; % Not sure about 4 and 7, which are very fast ones

%% Flip inversed channels

infChanIndx = find(invChanInd);
if ~isempty(infChanIndx)
  for iCh = 1:length(infChanIndx)
    iChan = infChanIndx(iCh);
    tmp = digIn.onset{iChan};
    digIn.onset{iChan} = digIn.offset{iChan};
    digIn.offset{iChan} = tmp;
  end
end

%% How to define trials?
% We start in environment 1. Can we add a toggle on start? Now the signal is inverted
% We can set the start of the first env1 to sample 1

if digIn.onset{11}(1) > digIn.offset{11}(1)
  env1St = [1; digIn.onset{11}];
else
  env1St = digIn.onset{11};
end

env2St = digIn.onset{12};
env3St = digIn.onset{16}; % 13

% env1Nd = digIn.offset{11};
% env2Nd = digIn.offset{12};
% env3Nd = digIn.offset{16};
% envAllNd = sort([env1Nd; env2Nd; env3Nd]);

trialAid = [ones(length(env1St),1); ones(length(env2St),1)*2; ones(length(env3St),1)*3];
[envAllSt, sortOrder] = sort([env1St; env2St; env3St]);
trialEnvId = trialAid(sortOrder);

% Trials should have a certain length
% vrDist = double(packet.longVar(:,2));
% diff(vrDist(envAllSt))
% These distances do not make sense

% In this particular case we can say that a trial needs a reward zone
rewZSt = digIn.onset{10}; % 9
edges = [envAllSt; inf];
rewZSt(rewZSt<edges(1)) = [];
[hasRewZone, trlIndx] = histc(rewZSt, edges);
trialSt = envAllSt(trlIndx);
trialId = trialEnvId(trlIndx);
nTrials = length(trialSt);

% plot(trialSt,ones(length(trialSt),1)*2.5,'ro')

%% Now we can start a trlMat
% How many colums do we want?

% 1 env onset
% 2 env id
% 3 sound onset
% 4 sound offset
% 5 tunnel1 onset
% 6 rewZone onset
% 7 reward delivery
% 8 tunnel2 onset
% 9 tunnel2 offset
% 10 trial duration

trialInfo = nan(nTrials,10);
trialInfoLabels = cell(1,10);
trialInfo(:,1) = trialSt;
trialInfoLabels{1} = 'environmentStart';
trialInfo(:,2) = trialId;
trialInfoLabels{2} = 'environmentID';

%% Tunnel 1 onset

tun1St = digIn.onset{14};
edges = [trialSt; inf];
tun1St(tun1St < edges(1)) = [];
[bin, indx] = histc(tun1St, edges);
[bl1 bl2] = unique(indx);
trialInfo(bl1,5) = tun1St(bl2);
trialInfoLabels{5} = 'tunnel1Start';

%% Sound

% Onset
soundOn = digIn.onset{8};
tmp = trialInfo(:,[1 5]);
edges = [sort(tmp(:)); inf];
soundOn(soundOn < edges(1)) = [];
[bin, indx] = histc(soundOn, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trialInfo(trlNum,3) = soundOn(isOdd);
trialInfoLabels{3} = 'soundOnset';

% Offset
soundOff = digIn.offset{8};
soundOff(soundOff < edges(1)) = [];
[bin, indx] = histc(soundOff, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trialInfo(trlNum,4) = soundOff(isOdd);
trialInfoLabels{4} = 'soundOffset';

%% Reward Zone onset

rewZSt = digIn.onset{10};
edges = [trialSt; inf];
rewZSt(rewZSt < edges(1)) = [];
[bin, indx] = histc(rewZSt, edges);
[bl1 bl2] = unique(indx);
trialInfo(bl1,6) = rewZSt(bl2);
trialInfoLabels{6} = 'rewardZoneStart';

%% Tunnel 2 onset

tun2St = digIn.onset{15};
edges = [trialSt; inf];
tun2St(tun2St < edges(1)) = [];
[bin, indx] = histc(tun2St, edges);
[bl1 bl2] = unique(indx);
trialInfo(bl1,8) = tun2St(bl2);
trialInfoLabels{8} = 'tunnel2Start';

%% Tunnel 2 offset
% The normal way does not work because the events are so close to env onset
% We can take the matching offset for the onset?

tun2Nd = digIn.offset{15};
edges = [trialInfo(:,8); inf];
tun2Nd(tun2Nd < edges(1)) = [];
[bin, indx] = histc(tun2Nd, edges);
[bl1 bl2] = unique(indx);
trialInfo(bl1,9) = tun2Nd(bl2);
trialInfoLabels{9} = 'tunnel2End';

%% Reward delivery (1rst)

rewDel = digOut.onset{2};
tmp = trialInfo(:,[6 8]);
edges = [sort(tmp(:)); inf];
rewDel(rewDel < edges(1)) = [];
[bin, indx] = histc(rewDel, edges);
% Get the odd indexes because they are within the environment period
[bl1 bl2] = unique(indx);
indx = indx(bl2); % remove the double, keep the first
rewDel = rewDel(bl2);
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trialInfo(trlNum,7) = rewDel(isOdd);
trialInfoLabels{7} = 'rewardDelivery'; % WARNING there can be more this is just the first one in the reward zone

%% Trial Duration
% Do we want the total duration or the duration of a specific part?
% Lets start lazy

trialInfo(:,10) = trialInfo(:,9) - trialInfo(:,1);
trialInfoLabels{10} = 'trialDuration';

%% What is the latency of the transition events?
% env - tunnel1 - rewzone - tunnel2

tun1St = digIn.onset{13};
tun1Nd = digIn.offset{13};
tun2St = digIn.onset{14};
tun2Nd = digIn.offset{14};
rewZSt = digIn.onset{9};
rewZNd = digIn.offset{9};

disp(envAllNd - tun1St)
disp(tun1Nd(1:end-1) - rewZSt)
disp(rewZNd - tun2St(1:end-1))
disp(tun2Nd - envAllSt(2:end-2))
% These latencies are actually quite spectacularly good



%% For the 2nd segment 600000 - 1400000

% invertedChannels: 11 13 14


packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded.mat');

%% In this session Unity was stopped and restarted twice. This creates problems. For now just take the first 6*10^5 values

packet.digitalIn([1:600000 1400001:end],:) = [];
packet.digitalOut([1:600000 1400001:end],:) = [];
packet.longVar([1:600000 1400001:end],:) = [];
packet.packetNums([1:600000 1400001:end]) = [];
% packet.startTS([1:600000 1400001:end]) = [];

offset2 = 600000;

%% Digital Inputs

% Load file
tempInput = double(packet.digitalIn);
% sum(digitalIn)
goodOrder=[8:-1:1 16:-1:9];
nChan = size(tempInput,2);
digitalIn = nan(size(tempInput));
for iCh = 1:nChan
    iChan = goodOrder(iCh);
    digitalIn(:,iChan) = tempInput(:,iCh);
end

nChan = size(digitalIn,2);
for iChan = 1:nChan
  dChan = diff(digitalIn(:,iChan));
  digIn.onset{iChan} = find(dChan == 1) + 1;
  digIn.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% what is in where?
% which have stuff: 1vid1 2vid2 3iti 8? 9scanClock 10? 13belt? 14belt? 
digIn.labels = {'empty','empty','wheelA','wheelB', ...
  'wheelC','IRcamera','sound','scanner', ...
  'reward_zone','environment1','environment2','environment3', ...
  'tunnel1','tunnel2','empty','empty'};


%% Plot stuff

nPack = size(digitalIn,1);
figure; hold on
for iChan = [7 9:nChan]
%   plot(offset:offset+nPack-1,digitalIn(:,iChan)*iChan)
  plot(digitalIn(:,iChan)*iChan)
end

legend('sound','rewardZone','env1','env2','env3','tunnel1','tunnel2')

%% Digital Outputs
% Load file
tempOutput = double(packet.digitalOut);
% sum(digitalIn)
goodOrder= 8:-1:1;
nChan = size(tempOutput,2);
digitalOut = nan(size(tempOutput));
for iCh = 1:nChan
    iChan = goodOrder(iCh);
    digitalOut(:,iChan) = tempOutput(:,iCh);
end

% Load file
% digitalOut = double(packet.digitalOut);
% sum(digitalIn)

nChan = size(digitalOut,2);
for iChan = 1:nChan
  dChan = diff(digitalOut(:,iChan));
  digOut.onset{iChan} = find(dChan == 1) + 1;
  digOut.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% which channels have stuff 4sync?
digOut.labels = {'empty','valve','empty','empty', ...
  'barcode','empty','lick','empty'};

%% Find inverted channels
% This fails on the short segment of data, but should still be ok for
% larger sessions. We Can look for a way to improve

nChan = size(digitalIn,2);
invChanInd = false(1,nChan);
for iChan = 1:nChan
  nOn = length(digIn.onset{iChan});
  nOff = length(digIn.offset{iChan});
  nEv = min([nOn nOff]);
  if nEv == 0; continue; end
  medFor = median(digIn.offset{iChan}(1:nEv) - digIn.onset{iChan}(1:nEv));
  medInf = median(digIn.onset{iChan}(2:nEv) - digIn.offset{iChan}(1:nEv-1));
  if medInf < medFor
    invChanInd(iChan) = true;
  end
end

% QUICK FIX FOR NOW
invChanInd([11 14]) = true;


%% Flip inversed channels

infChanIndx = find(invChanInd);
if ~isempty(infChanIndx)
  for iCh = 1:length(infChanIndx)
    iChan = infChanIndx(iCh);
    tmp = digIn.onset{iChan};
    digIn.onset{iChan} = digIn.offset{iChan};
    digIn.offset{iChan} = tmp;
  end
end

%% How to define trials?
% We start in environment 1. Can we add a toggle on start? Now the signal is inverted
% We can set the start of the first env1 to sample 1

env1St = digIn.onset{10};
env2St = digIn.onset{11};
env3St = digIn.onset{12};

env1Nd = digIn.offset{10};
env2Nd = digIn.offset{11};
env3Nd = digIn.offset{12};
envAllNd = sort([env1Nd; env2Nd; env3Nd]);

trialAid = [ones(length(env1St),1); ones(length(env2St),1)*2; ones(length(env3St),1)*3];
[envAllSt, sortOrder] = sort([env1St; env2St; env3St]);
trialEnvId = trialAid(sortOrder);

% Trials should have a certain length
vrDist = double(packet.longVar(:,2));
% diff(vrDist(envAllSt))
% These distances do not make sense

% In this particular case we can say that a trial need a reward zone
rewZSt = digIn.onset{9};
edges = [envAllSt; inf];
rewZSt(rewZSt<edges(1)) = [];
[hasRewZone, trlIndx] = histc(rewZSt, edges);
trialSt = envAllSt(trlIndx);
trialId = trialEnvId(trlIndx);
nTrials = length(trialSt);

%% Now we can start a trlMat
% How many colums do we want?

% 1 env onset
% 2 env id
% 3 sound onset
% 4 sound offset
% 5 tunnel1 onset
% 6 rewZone onset
% 7 reward delivery
% 8 tunnel2 onset
% 9 tunnel2 offset
% 10 trial duration

trlMat2 = nan(nTrials,10);
trialInfoLabels = cell(1,10);
trlMat2(:,1) = trialSt;
trialInfoLabels{1} = 'environmentStart';
trlMat2(:,2) = trialId;
trialInfoLabels{2} = 'environmentID';

%% Tunnel 1 onset

tun1St = digIn.onset{13};
edges = [trialSt; inf];
tun1St(tun1St < edges(1)) = [];
[bin, indx] = histc(tun1St, edges);
[bl1 bl2] = unique(indx);
trlMat2(bl1,5) = tun1St(bl2);
trialInfoLabels{5} = 'tunnel1Start';

%% Sound

% Onset
soundOn = digIn.onset{7};
tmp = trlMat2(:,[1 5]);
edges = [sort(tmp(:)); inf];
soundOn(soundOn < edges(1)) = [];
[bin, indx] = histc(soundOn, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat2(trlNum,3) = soundOn(isOdd);
trialInfoLabels{3} = 'soundOnset';

% Offset
soundOff = digIn.offset{7};
soundOff(soundOff < edges(1)) = [];
[bin, indx] = histc(soundOff, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat2(trlNum,4) = soundOff(isOdd);
trialInfoLabels{4} = 'soundOffset';

%% Reward Zone onset
% We can also start using more tight periods

rewZSt = digIn.onset{9};
edges = [trialSt; inf];
rewZSt(rewZSt < edges(1)) = [];
[bin, indx] = histc(rewZSt, edges);
[bl1 bl2] = unique(indx);
trlMat2(bl1,6) = rewZSt(bl2);
trialInfoLabels{6} = 'rewardZoneStart';

%% Tunnel 2 onset

tun2St = digIn.onset{14};
edges = [trialSt; inf];
tun2St(tun2St < edges(1)) = [];
[bin, indx] = histc(tun2St, edges);
[bl1 bl2] = unique(indx);
trlMat2(bl1,8) = tun2St(bl2);
trialInfoLabels{8} = 'tunnel2Start';

%% Tunnel 2 offset
% The normal way does not work because the events are so close to env onset
% We can take the matching offset for the onset?

tun2Nd = digIn.offset{14};
edges = [trlMat2(:,8); inf];
tun2Nd(tun2Nd < edges(1)) = [];
[bin, indx] = histc(tun2Nd, edges);
[bl1 bl2] = unique(indx);
trlMat2(bl1,9) = tun2Nd(bl2);
trialInfoLabels{9} = 'tunnel2End';

%% Reward delivery (1rst)

rewDel = digOut.onset{2};
tmp = trlMat2(:,[6 8]);
edges = [sort(tmp(:)); inf];
[bin, indx] = histc(rewDel, edges);
% Get the odd indexes because they are within the environment period
[bl1 bl2] = unique(indx);
indx = indx(bl2); % remove the double, keep the first
rewDel = rewDel(bl2);
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat2(trlNum,7) = rewDel(isOdd);
trialInfoLabels{7} = 'rewardDelivery'; % WARNING there can be more this is just the first one in the reward zone

%% Trial Duration
% Do we want the total duration or the duration of a specific part?
% Lets start lazy

trlMat2(:,10) = trlMat2(:,9) - trlMat2(:,1);
trialInfoLabels{10} = 'trialDuration';

%% What is the latency of the transition events?
% env - tunnel1 - rewzone - tunnel2

tun1St = digIn.onset{13};
tun1Nd = digIn.offset{13};
tun2St = digIn.onset{14};
tun2Nd = digIn.offset{14};
rewZSt = digIn.onset{9};
rewZNd = digIn.offset{9};

disp(envAllNd - tun1St)
disp(tun1Nd - rewZSt)
disp(rewZNd - tun2St)
disp(tun2Nd - envAllSt)
% These latencies are actually quite spectacularly good


%% For the 3rd segment 1600000 - 2100000

% invertedChannels: 11

packet = load('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\20220513-165536_677_decoded.mat');

%% In this session Unity was stopped and restarted twice. This creates problems. For now just take the first 6*10^5 values

packet.digitalIn(1:1600001,:) = [];
packet.digitalOut(1:1600001,:) = [];
packet.longVar(1:1600001) = [];
packet.packetNums(1:1600001) = [];

offset3 = 1600000;

%% Digital Inputs

% Load file
tempInput = double(packet.digitalIn);
% sum(digitalIn)
goodOrder=[8:-1:1 16:-1:9];
nChan = size(tempInput,2);
digitalIn = nan(size(tempInput));
for iCh = 1:nChan
    iChan = goodOrder(iCh);
    digitalIn(:,iChan) = tempInput(:,iCh);
end

nChan = size(digitalIn,2);
for iChan = 1:nChan
  dChan = diff(digitalIn(:,iChan));
  digIn.onset{iChan} = find(dChan == 1) + 1;
  digIn.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% what is in where?
% which have stuff: 1vid1 2vid2 3iti 8? 9scanClock 10? 13belt? 14belt? 
digIn.labels = {'empty','empty','wheelA','wheelB', ...
  'wheelC','IRcamera','sound','scanner', ...
  'reward_zone','environment1','environment2','environment3', ...
  'tunnel1','tunnel2','empty','empty'};


%% Plot stuff

nPack = size(digitalIn,1);
figure; hold on
for iChan = [7 9:nChan]
%   plot(offset:offset+nPack-1,digitalIn(:,iChan)*iChan)
  plot(digitalIn(:,iChan)*iChan)
end

legend('sound','rewardZone','env1','env2','env3','tunnel1','tunnel2')

%% Digital Outputs
% Load file
tempOutput = double(packet.digitalOut);
% sum(digitalIn)
goodOrder= 8:-1:1;
nChan = size(tempOutput,2);
digitalOut = nan(size(tempOutput));
for iCh = 1:nChan
    iChan = goodOrder(iCh);
    digitalOut(:,iChan) = tempOutput(:,iCh);
end

% Load file
% digitalOut = double(packet.digitalOut);
% sum(digitalIn)

nChan = size(digitalOut,2);
for iChan = 1:nChan
  dChan = diff(digitalOut(:,iChan));
  digOut.onset{iChan} = find(dChan == 1) + 1;
  digOut.offset{iChan} = find(dChan == -1);
  % we should add a boundry check
end
% which channels have stuff 4sync?
digOut.labels = {'empty','valve','empty','empty', ...
  'barcode','empty','lick','empty'};

%% Find inverted channels
% This fails on the short segment of data, but should still be ok for
% larger sessions. We Can look for a way to improve

nChan = size(digitalIn,2);
invChanInd = false(1,nChan);
for iChan = 1:nChan
  nOn = length(digIn.onset{iChan});
  nOff = length(digIn.offset{iChan});
  nEv = min([nOn nOff]);
  if nEv == 0; continue; end
  medFor = median(digIn.offset{iChan}(1:nEv) - digIn.onset{iChan}(1:nEv));
  medInf = median(digIn.onset{iChan}(2:nEv) - digIn.offset{iChan}(1:nEv-1));
  if medInf < medFor
    invChanInd(iChan) = true;
  end
end

% QUICK FIX FOR NOW
invChanInd(10) = false;
invChanInd(11) = true;


%% Flip inversed channels

infChanIndx = find(invChanInd);
if ~isempty(infChanIndx)
  for iCh = 1:length(infChanIndx)
    iChan = infChanIndx(iCh);
    tmp = digIn.onset{iChan};
    digIn.onset{iChan} = digIn.offset{iChan};
    digIn.offset{iChan} = tmp;
  end
end

%% How to define trials?
% We start in environment 1. Can we add a toggle on start? Now the signal is inverted
% We can set the start of the first env1 to sample 1

env1St = digIn.onset{10};
env2St = digIn.onset{11};
env3St = digIn.onset{12};

env1Nd = digIn.offset{10};
env2Nd = digIn.offset{11};
env3Nd = digIn.offset{12};
envAllNd = sort([env1Nd; env2Nd; env3Nd]);

trialAid = [ones(length(env1St),1); ones(length(env2St),1)*2; ones(length(env3St),1)*3];
[envAllSt, sortOrder] = sort([env1St; env2St; env3St]);
trialEnvId = trialAid(sortOrder);

% Trials should have a certain length
vrDist = double(packet.longVar(:,2));
% diff(vrDist(envAllSt))
% These distances do not make sense

% In this particular case we can say that a trial need a reward zone
rewZSt = digIn.onset{9};
edges = [envAllSt; inf];
rewZSt(rewZSt<edges(1)) = [];
[hasRewZone, trlIndx] = histc(rewZSt, edges);
trialSt = envAllSt(trlIndx);
trialId = trialEnvId(trlIndx);
nTrials = length(trialSt);

%% Now we can start a trlMat
% How many colums do we want?

% 1 env onset
% 2 env id
% 3 sound onset
% 4 sound offset
% 5 tunnel1 onset
% 6 rewZone onset
% 7 reward delivery
% 8 tunnel2 onset
% 9 tunnel2 offset
% 10 trial duration

trlMat3 = nan(nTrials,10);
trialInfoLabels = cell(1,10);
trlMat3(:,1) = trialSt;
trialInfoLabels{1} = 'environmentStart';
trlMat3(:,2) = trialId;
trialInfoLabels{2} = 'environmentID';

%% Tunnel 1 onset

tun1St = digIn.onset{13};
edges = [trialSt; inf];
tun1St(tun1St < edges(1)) = [];
[bin, indx] = histc(tun1St, edges);
[bl1 bl2] = unique(indx);
trlMat3(bl1,5) = tun1St(bl2);
trialInfoLabels{5} = 'tunnel1Start';

%% Sound

% Onset
soundOn = digIn.onset{7};
tmp = trlMat3(:,[1 5]);
edges = [sort(tmp(:)); inf];
soundOn(soundOn < edges(1)) = [];
[bin, indx] = histc(soundOn, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat3(trlNum,3) = soundOn(isOdd);
trialInfoLabels{3} = 'soundOnset';

% Offset
soundOff = digIn.offset{7};
soundOff(soundOff < edges(1)) = [];
[bin, indx] = histc(soundOff, edges);
% Get the odd indexes because they are within the environment period
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat3(trlNum,4) = soundOff(isOdd);
trialInfoLabels{4} = 'soundOffset';

%% Reward Zone onset
% We can also start using more tight periods

rewZSt = digIn.onset{9};
edges = [trialSt; inf];
rewZSt(rewZSt < edges(1)) = [];
[bin, indx] = histc(rewZSt, edges);
[bl1 bl2] = unique(indx);
trlMat3(bl1,6) = rewZSt(bl2);
trialInfoLabels{6} = 'rewardZoneStart';

%% Tunnel 2 onset

tun2St = digIn.onset{14};
edges = [trialSt; inf];
tun2St(tun2St < edges(1)) = [];
[bin, indx] = histc(tun2St, edges);
[bl1 bl2] = unique(indx);
trlMat3(bl1,8) = tun2St(bl2);
trialInfoLabels{8} = 'tunnel2Start';

%% Tunnel 2 offset
% The normal way does not work because the events are so close to env onset
% We can take the matching offset for the onset?

tun2Nd = digIn.offset{14};
edges = [trlMat3(:,8); inf];
tun2Nd(tun2Nd < edges(1)) = [];
[bin, indx] = histc(tun2Nd, edges);
[bl1 bl2] = unique(indx);
trlMat3(bl1,9) = tun2Nd(bl2);
trialInfoLabels{9} = 'tunnel2End';

%% Reward delivery (1rst)

rewDel = digOut.onset{2};
tmp = trlMat3(:,[6 8]);
edges = [sort(tmp(:)); inf];
[bin, indx] = histc(rewDel, edges);
% Get the odd indexes because they are within the environment period
[bl1 bl2] = unique(indx);
indx = indx(bl2); % remove the double, keep the first
rewDel = rewDel(bl2);
isOdd = rem(indx, 2) == 1;
trlNum = ceil((indx(isOdd))/2);
trlMat3(trlNum,7) = rewDel(isOdd);
trialInfoLabels{7} = 'rewardDelivery'; % WARNING there can be more this is just the first one in the reward zone

%% Trial Duration
% Do we want the total duration or the duration of a specific part?
% Lets start lazy

trlMat3(:,10) = trlMat3(:,9) - trlMat3(:,1);
trialInfoLabels{10} = 'trialDuration';

%% What is the latency of the transition events?
% env - tunnel1 - rewzone - tunnel2

tun1St = digIn.onset{13};
tun1Nd = digIn.offset{13};
tun2St = digIn.onset{14};
tun2Nd = digIn.offset{14};
rewZSt = digIn.onset{9};
rewZNd = digIn.offset{9};

disp(envAllNd - tun1St(2:end))
disp(tun1Nd - rewZSt(2:end))
disp(rewZNd(2:end) - tun2St)
disp(tun2Nd(2:end) - envAllSt)
% These latencies are actually quite spectacularly good


%% When our powers combine

trlMat2(:,[1 3:9]) = trlMat2(:,[1 3:9]) + offset2;
trlMat3(:,[1 3:9]) = trlMat3(:,[1 3:9]) + offset3;
trialInfo = [trialInfo; trlMat2; trlMat3];

%% Solve the 120 missing package thingy when unity is turned off

% packet.startTS are times in us, which include the skips, so we can just
% use the indexes from trlMat

rawTs = double(packet.startTS);
firstPackage = rawTs(1) - 1000; % 1ms offset to make the count start at 1
trlMatOld = trialInfo;
trialInfoTs = nan(size(trialInfo));
trialInfoMs = nan(size(trialInfo));
nEv = size(trialInfo,2);
for iEv = 1:nEv
  if iEv == 2
    trialInfoTs(:,iEv) = trialInfo(:,iEv);
    trialInfoMs(:,iEv) = trialInfo(:,iEv);
    continue;
  elseif iEv == 10
    noNanInd = ~isnan(trialInfo(:,iEv));
    trialInfoTs(noNanInd,iEv) = trialInfoTs(noNanInd,9) - trialInfoTs(noNanInd,1);
    trialInfoMs(noNanInd,iEv) = trialInfoMs(noNanInd,9) - trialInfoMs(noNanInd,1);
    continue;
  end
  noNanInd = ~isnan(trialInfo(:,iEv));
  trialInfoTs(noNanInd,iEv) = rawTs(trialInfo(noNanInd,iEv));
  trialInfoMs(noNanInd,iEv) = (rawTs(trialInfo(noNanInd,iEv))-firstPackage)/10^3;
end

%% Save the data

save('C:\Users\Jeroen\Documents\Nijmegen\2P\2PData\trialInfo_20220513','trialInfo', ...
            'trialInfoTs','trialInfoMs','trialInfoLabels')


















































