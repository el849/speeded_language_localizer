function evlab_langloc_speeded(subj_id, set, run)

% Created By: Terri Scott (tlscott@mit.edu) 4/8/13, 
% edits by Greta Tuckute, summer 2021. edits by Aalok Sathe, 2022.

% This is a language localizer experiment with two conditions - sentences
% and nonwords. In each trial, 12 word-long sequences will be displayed one
% word at a time followed by a prompt for the subject to push a button, 
% just to make sure they are paying attention. 
% The subject should be instructed to read each English word or nonword and
% press a button when then image of a hand pressing a button is displayed.
% Emphasis should be placed on paying attention to reading each sequence
% and not to be stressed out if the sequence seems fast. 

% The trial timings are as follows: 100 ms of blank screen, 200 ms *
% 12 words for 2400 ms of stimuli, 400 ms of the button press image, and
% 100 ms of blank screen. The entire trial lasts for 3000 ms. The subject's 
% button press for a given trial will
% be recorded if it occurs after the button press image and before the
% same image of the subsequent trial.

% 3 trials of a given condition will be grouped into a block. A run will
% consist of 16 blocks in this sequence: Fix B1 B2 B3 B4 Fix B5 B6 B7 B8
% Fix B9 B10 B11 B12 Fix B13 B14 B15 B16 Fix. Each fixation period will
% last 14000 ms. Each run is 3 minutes, 34 s (214 s). 107 TRs, assuming TR=2s.

% Argument definitions:

% subj_id: Should be a string designating the subject.

% run: The localizer is meant to be run twice. The run is the counter-balance number, 
% and should have the value 1 or 2 delineating the first or second run of the experiment. 
% The sequence of conditions are:
% run = 1 : SNNS - NSNS - SNSN - NSSN
% run = 2 : NSSN - SNSN - NSNS - SNNS
% each 'S' or 'N' here is three 12-word-long items.

% A structure 'subj_data' is created and saved to the pwd unless otherwise
% specified. It will be saved as a .mat in the format:
% evlab_langloc_speeded_2022_<subj_id>_fmri_run<run#>_data.mat. 
% If the output file already exists, it will be suffixed with the repeat #.

%% Parameters to change
Screen('Preference', 'SkipSyncTests', 1);
% Screen('Preference','SyncTestSettings',.0004); % Run it with strict timing settings for testing
Screen('Preference', 'DefaultFontSize', 200);
KbName('UnifyKeyNames');
screensAll = Screen('Screens');
screenNum = max(screensAll); % Which screen you want to use. "1" is external monitor, "0" is this screen.
my_key = '1!'; % What key gives a response.
my_trigger = '=+'; % What key triggers the script from the scanner.
do_suppress_warnings = 1; % You don't need to do this but I don't like the warning screen at the beginning.
addpath([pwd filesep 'func']);
DATA_DIR = [pwd filesep 'data']; % Where the subj_data will be saved.
STIM_DIR = [pwd filesep 'new_stim']; % Where all the stimuli are.
stim_font_size = 100;
                      

%% Define the output file 
file_to_save = ['evlab_langloc_speeded_2022_' subj_id '_fmri_run' num2str(run) '_set' num2str(set) '_data.mat']; 

% Error message if data file already exists.
if exist([DATA_DIR filesep file_to_save],'file')

    all_files = dir([DATA_DIR filesep file_to_save]);
    all_files = {all_files.name};
    
    file_to_save = ['evlab_langloc_speeded_2022_' subj_id '_fmri_run' num2str(run) '_set' num2str(set) '_repeat' num2str(length(all_files)) '_data.mat']; 
    
end

clear subj_data

% Experiment settings
num_of_trials = 48;
num_of_fix = 5;
word_time = 0.200;
trial_time = 12*word_time + 0.600;

%% Start experiment

% Choose which stimuli set to use.

    % stim = load([STIM_DIR filesep 'langloc_fmri_run' num2str(run) '_stim_set' num2str(set) '.mat']);  % under folder old_stim
    stim = load([STIM_DIR filesep 'speeded_langloc_2022_fmri_run' num2str(run) '_stim_set' num2str(set) '.mat']);
    stim = stim.stim;
    
% Load variables needed later. 

img=imread([pwd filesep 'images' filesep 'hand-press-button-4.jpeg'], 'JPG');

did_subj_respond = 0;
r_count = 1;

trial_times = zeros(48,1);
for i = 1:num_of_trials
    if ismember(i,[13 25 37])
        trial_times(i) = trial_times(i-1) + 14.000 + trial_time;
    elseif i == 1
        trial_times(i) = 0.000;
    else
        trial_times(i) = trial_times(i-1) + trial_time;
    end
end

subj_data.id = subj_id;
subj_data.did_respond = zeros(num_of_trials,1);
subj_data.probe_onset = zeros(num_of_trials,1);
subj_data.probe_response = zeros(num_of_trials,1);
subj_data.trial_onsets = zeros(num_of_trials,1);
subj_data.fix_onsets = zeros(num_of_fix,1);
subj_data.condition = cell(num_of_trials,1);
subj_data.stim = cell(num_of_trials,1);
subj_data.run = run;

% Update the conditions / stim in output structure 
for i = 1 : num_of_trials
    subj_data.stim{i}=strjoin(stim(i,2:13));
    subj_data.condition(i)=stim(i,14);
end

% Save all data to current folder.
save([DATA_DIR filesep file_to_save], 'subj_data');

% Screen preferences

% Setting this preference to 1 suppresses the printout of warnings.
oldEnableFlag = Screen('Preference', 'SuppressAllWarnings', do_suppress_warnings);


% Open screen.

[wPtr,~]=Screen('OpenWindow',screenNum,1);

white=WhiteIndex(wPtr);
scrColor = white;
Screen('FillRect',wPtr,scrColor);
Screen(wPtr, 'Flip');

HideCursor;

Screen('Preference', 'TextRenderer', 1);
Screen('TextFont', wPtr, '-misc-fixed-medium-o-normal--13-120-75-75-c-70-iso8859-1');
Screen('TextSize', wPtr , stim_font_size);
DrawFormattedText(wPtr,'Waiting for scanner...','center','center');
Screen(wPtr, 'Flip');

% Pre-draw button press image
textureIndex=Screen('MakeTexture', wPtr, double(img));

% Get trigger from scanner.

TRIGGER_KEY = [KbName('=+'), KbName('+')]; %,KbName('=')]; % ** AK changed this 081612 % if this doesn't work, change to '=+'
while 1
    [keyIsDown, seconds, keyCode] = KbCheck(-3);

    if ismember(find(keyCode,1),TRIGGER_KEY) %keyCode(KbName(TRIGGER_KEY))
        break
    end
    WaitSecs('YieldSecs', 0.0001); % Wait for yieldInterval to prevent system overload.
end

subj_data.run_onset = GetSecs;

%% Runs

try

% Fixation

%Screen('TextSize', wPtr , 100); A.S. changed this to the below
Screen('TextSize', wPtr , stim_font_size);
DrawFormattedText(wPtr,'+','center','center');
Screen(wPtr, 'Flip');

subj_data.fix_onsets(1) = GetSecs;

% Calculate trial onsets:

subj_data.i_trial_onsets = (subj_data.run_onset+14.000) + trial_times;

while GetSecs<14.000+subj_data.run_onset
    WaitSecs('YieldSecs', 0.0001);
end

% Start trials

for i = 1:num_of_trials
    
    subj_data.trial_onsets(i) = GetSecs;
    stim_seq = stim(i,2:13);
    
    % White screen for 100 ms
    
    white=WhiteIndex(wPtr);
    Screen('FillRect',wPtr,scrColor);
    Screen(wPtr, 'Flip');
    
    while GetSecs<0.100+subj_data.i_trial_onsets(i)
        if did_subj_respond == 0
            did_subj_respond = getKeyResponse;           
        end
        WaitSecs('YieldSecs', 0.0001);
    end
    
    % Sequence presentation 12 * 450 ms
    
    Screen('TextSize', wPtr , stim_font_size);
    
    for j = 1:12
       
        DrawFormattedText(wPtr,stim_seq{j},'center','center');
        Screen(wPtr, 'Flip');
        
        while GetSecs<word_time*j + 0.100 + subj_data.i_trial_onsets(i)
            if did_subj_respond == 0
                did_subj_respond = getKeyResponse;           
            end
            WaitSecs('YieldSecs', 0.0001);
        end
    end
    
    % Present image for word_time (200 ms)
    
    subj_data.did_respond(r_count) = did_subj_respond;
    
    if i ~= 1
         r_count = r_count + 1;
    end
    
    did_subj_respond = 0;
         
    Screen('DrawTexture', wPtr, textureIndex);
    Screen(wPtr, 'Flip');
    
    subj_data.probe_onset(i) = GetSecs;
    
    while GetSecs<0.400+(12*word_time)+0.100+subj_data.i_trial_onsets(i)
        if did_subj_respond == 0
            did_subj_respond = getKeyResponse;
        end
        WaitSecs('YieldSecs', 0.0001);
    end
    
    % White screen for 100 ms
    
    Screen('FillRect',wPtr,scrColor);
    Screen(wPtr, 'Flip');
    
    while GetSecs<0.100+0.400+(12*word_time)+0.100+subj_data.i_trial_onsets(i)
        if did_subj_respond == 0
            did_subj_respond = getKeyResponse;
        end
        WaitSecs('YieldSecs', 0.0001);
    end
    
    if ismember(i,[12,24,36,48]) % Fixation occurs after every 12 trials or 4 blocks
        
        % Fixation
        
        %Screen('TextSize', wPtr , 100); % AS 20220331
        Screen('TextSize', wPtr , stim_font_size);
        DrawFormattedText(wPtr,'+','center','center');
        Screen(wPtr, 'Flip');
        
        subj_data.fix_onsets(i/12+1) = GetSecs;
        
        % 14s is fixation time; 100ms blank; 400ms is the button press img
        while GetSecs<14.000+0.100+0.400+(12*word_time)+0.100+subj_data.i_trial_onsets(i)
            if did_subj_respond == 0
                did_subj_respond = getKeyResponse;
            end
            WaitSecs('YieldSecs', 0.0001);
        end
        
    end   
end

subj_data.runtime = GetSecs - subj_data.run_onset;

subj_data.did_respond(r_count) = did_subj_respond;

Screen('CloseAll');
ShowCursor

% Get reaction time.
subj_data.rt = zeros(num_of_trials,1);
responses = find(subj_data.did_respond);
subj_data.rt(responses) = subj_data.probe_response(responses) - subj_data.probe_onset(responses);

% Save all data to current folder.
save([DATA_DIR filesep file_to_save], 'subj_data');

catch err
    
    subj_data.did_respond(r_count) = did_subj_respond;
    
    % Get reaction time.
    subj_data.rt = zeros(num_of_trials,1);
    responses = find(subj_data.did_respond);
    subj_data.rt(responses) = subj_data.probe_response(responses) - subj_data.probe_onset(responses);
    
    subj_data.set=set;
    
    % Save all data to current folder.
    save([DATA_DIR filesep file_to_save], 'subj_data','err');
    
    % Fixation with red cross after trials end
    Screen('TextSize', wPtr , stim_font_size);
    DrawFormattedText(wPtr,'+','center','center', [255 0 0]);
    Screen(wPtr, 'Flip');
    
    Screen('CloseAll');
    ShowCursor
end

% At the end of your code, it is a good idea to restore the old level.
Screen('Preference','SuppressAllWarnings',oldEnableFlag);



%%
% % % % % % % %
% SUBFUNCTION %
% % % % % % % %

    function out = getKeyResponse
        KEY1=KbName(my_key);
        
        [keyIsDown,x,keyCode]=KbCheck;
        if keyIsDown
            response=find(keyCode);
            if response==KEY1
                out = 1;
                subj_data.probe_response(r_count) = x;
            else
                out = 0;
            end
        else
            out = 0;
        end
    end
end
