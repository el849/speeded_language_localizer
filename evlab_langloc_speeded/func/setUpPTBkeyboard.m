function [keyNames, Button, kbIdx, oldKeyboardPrefs] = setUpPTBkeyboard()
%setUpPTBkeyboard
%   returns
%   keyNames: cell array where each element is a string that indicates how
%             PTB refers to each key
%
%   Button: a struct that contains useful keys
%           Button.trigger - keys that possibly trigger the scanner
%           Button.escape  - key you can press at any time to end the script
%
%   kbIdx: index of keyboard for capturing button box input
%
%   oldKeyboardPrefs: you should restore these at the end of the main script

	KbName('UnifyKeyNames');
    keyNames = KbName('KeyNames');
    
    %Possible triggers
    Button.trigger = [KbName('=+') KbName('+')];
    
    %Escape key
    Button.escape = KbName('Escape');
    
    %Hand Icon response
    Button.one = [KbName('1') KbName('1!')];
    Button.two = [KbName('2') KbName('2@')];
    
    
    %Restrict KbCheck to only check for keys we want to look for
    enableKeys = [Button.trigger, Button.escape, Button.one, Button.two];
    oldKeyboardPrefs.enableKeys = RestrictKeysForKbCheck(enableKeys);
    
    
    %kbIdx (index of keyboard for capturing button box input)
	if IsWin || IsLinux
        kbIdx = 0;
    else
        devices = PsychHID('devices');
        kbIdx = [];
        for dev = 1:length(devices)
            if strcmp(devices(dev).usageName, 'Keyboard')
                kbIdx = [kbIdx dev];
            end
        end
	end
end

