clear all
strList = {}; %this will come out with three dates where both hemispheres are 'right'
baseFolder = '/Volumes/datalake/aDBS-49155/preprocessed-new';


patientD = dir(fullfile(baseFolder, 'aDBS*'));

patientD = patientD(~strcmp({patientD.name}, 'aDBS003'));

StimParams = struct();

for p = 1:length(patientD)
    clear patientName
    patientPath = fullfile(patientD(p).folder, patientD(p).name);
    patientName = patientD(p).name;
    patientName = replace(patientName,'aDBS','x');

   
    dateD = dir(patientPath);

    dateD = dateD(~ismember({dateD.name}, {'.', '..'}));
    dateD = dateD([dateD.isdir]);

    for dI = 1:length(dateD)
        clear dateName
        datePath = fullfile(dateD(dI).folder, dateD(dI).name);
        dateName = dateD(dI).name;
        dateName = ['d',replace(dateName,'-','_')];



       
        taskD = dir(datePath);

        taskD = taskD(~ismember({taskD.name}, {'.', '..'}));
        taskD = taskD([taskD.isdir]);
        taskD = taskD(contains({taskD.name}, 'resting', 'IgnoreCase', true));

        if isempty(taskD)
            continue
        end


        for tI = 1:length(taskD)
            clear condtype
            if contains(taskD(tI).name,'off','IgnoreCase',true)
                condtype = 'xeDBSoff';

            elseif contains(taskD(tI).name,'3')
                condtype = 'xe3';

            elseif contains(taskD(tI).name,'2')
                condtype = 'xe2';

            elseif contains(taskD(tI).name,'1')
                condtype = 'xe1';

            elseif contains(taskD(tI).name,'resting')
                condtype = 'xe';
            
            else
                keyboard
                continue
            end


            taskPath = fullfile(taskD(tI).folder, taskD(tI).name);
            fileD = dir(taskPath);
            fileD = fileD(contains({fileD.name},'behav_lfp_v3.mat'));
            if length(fileD) > 1

                fileD = fileD(contains({fileD.name},'ephys'));

            end

            if isempty(fileD)
                continue
            end

            clear lfpData
            clear data
            load(fullfile(fileD.folder,fileD.name));
            clear leftCheck
            clear rightCheck

            if p < 4

                if ~exist('lfpData','var') || ~exist('data','var')

                    continue

                end
                StimParams.(patientName).(dateName).(condtype).BothHemi = lfpData.stimLogSettings;
                StimParams.(patientName).(dateName).(condtype).BehavStartUnix = data.behavior.behav_start_timestamp_unix;

            elseif p >= 4

                if ~exist('lfpData','var') || ~exist('data','var')

                    continue

                end


                if lfpData(1).metaData.subjectID(end) == 'L' && lfpData(2).metaData.subjectID(end) == 'R'

                    leftCheck = 1;
                    rightCheck = 2;



                elseif lfpData(2).metaData.subjectID(end) == 'L' && lfpData(1).metaData.subjectID(end) == 'R'
                    leftCheck = 2;
                    rightCheck = 1;


                end

                if ~exist('leftCheck','var') || ~exist('rightCheck','var')
                    strList{end+1} = strcat(patientName, dateName, condtype);

                    continue
                end

                StimParams.(patientName).(dateName).(condtype).left = lfpData(leftCheck).stimLogSettings;
                StimParams.(patientName).(dateName).(condtype).right = lfpData(rightCheck).stimLogSettings;
                StimParams.(patientName).(dateName).(condtype).BehavStartUnix = data.behavior.behav_start_timestamp_unix;

            end

        end

    end


    disp("Finished with a patient")
end

disp("FINISHED")
