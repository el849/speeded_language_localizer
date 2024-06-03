function runOnset = waitForTrigger(kbIdx, trigger, escape, window)
%waitForTrigger
%   waits for trigger from scanner before starting experiment
%
%   immediately flips screen, so something should already be already drawn
%   to wPtr
%
%   returns runOnset, the time that the trigger was sent

    while true
        %Check which buttons are pressed
        [~, runOnset, keyCode] = KbCheck(kbIdx);
        
        %Continue if trigger
        if any(keyCode(trigger))
            Screen('Flip',window);
            return
        end
        
        %Exit if escape key is pressed
        if any(keyCode(escape))
            error('escape!');
        end
    end
end