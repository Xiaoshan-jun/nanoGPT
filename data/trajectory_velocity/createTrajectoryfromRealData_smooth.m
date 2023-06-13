%fileID = fopen('datasets/real/train/train.txt','w');
%fileID = fopen('datasets/real/val/val.txt','w');
%fileID = fopen('datasets/real/vis/vis.txt','w');
%fileID = fopen('plot.txt','w');
agentcount = 0;
t = 0;
b = 0.1;
%fileID = fopen('realdata2_train_smooth.txt','w');
fileID = fopen('realdata2_val_smooth.txt','w');
for i = 0:80
    r = rem( i , 10 );
    if r == 8 
        continue; %for val
    end
    if r ~= 9
        continue; %for test
    end
    if i == 70
        continue; %bad data
    end
    filename = sprintf('realdata2/flightdata%d.csv', i);
    %fileID = fopen(filename2,'w');
    %fileID = fopen('plot.txt','w');
    T = readtable(filename);
    T = table2array(T);
    for k = 1:size(T)
        T(k, 1) = round(T(k, 1),1); %round all the time
    end
    %for each time, average the position
    a = size(T);
    tn = a(1);
%                 if t == 1
%                     h = 'new';
%                 elseif t <= 10
%                     h = 'past';
%                 else
%                     h = 'velocity';
%                 end
%                 historyx(t) = fullhistoryx(sorted_values(t));
%                 historyy(t) = fullhistoryy(sorted_values(t));
%                 historyz(t) = fullhistoryz(sorted_values(t));
%                 fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,historyx(t),historyy(t),historyz(t));
%             end floor(tn);
    half = tn/2;
    count = 0;
    sumx = 0;
    sumy = 0;
    sumz = 0;
    t = T(1,1);
    first = 1;
    fullhistoryx = [];
    fullhistoryy = [];
	fullhistoryz = [];
    %create fullhistory
    for k = 1:tn
        x = T(k, 2);
        y = T(k, 3);
        z = T(k,4);
        fullhistoryx(length(fullhistoryx)+1) = x;
        fullhistoryy(length(fullhistoryy)+1) = y;
        fullhistoryz(length(fullhistoryz)+1) = z;
    end
    %smooth full history
    l = 2;
    newfullhistoryx = [];
    newfullhistoryy = [];
    newfullhistoryz = [];
    for l = 2:length(fullhistoryx)
        if  abs(fullhistoryz(l) - fullhistoryz(l-1)) > 0.001 || abs(fullhistoryy(l) - fullhistoryy(l-1)) > 0.005 || abs(fullhistoryx(l) - fullhistoryx(l-1)) > 0.005
            newfullhistoryx(length(newfullhistoryx)+1) = fullhistoryx(l);
            newfullhistoryy(length(newfullhistoryy)+1) = fullhistoryy(l);
            newfullhistoryz(length(newfullhistoryz)+1) = fullhistoryz(l);
        end
    end
    fullhistoryx = newfullhistoryx;
    fullhistoryy = newfullhistoryy;
    fullhistoryz = newfullhistoryz;
    %pick 10 history from full history
    for l = 1:10
        historyx = [];
        historyy = [];
        historyz = [];
        
    % Generate random integers between lower_bound and upper_bound
        random_values = randi([1, length(fullhistoryx)], [1, 20]);

        % Sort the random values in ascending order
        sorted_values = sort(random_values);
%         for t2 = 0:9
%             for t = 1+t2:10+t2
%                 if t == 1 + t2
%                     h = 'new';
%                 elseif t <= 10 + t2
%                     h = 'past';
%                 else
%                     h = 'velocity';
%                 end
%                 historyx(t) = fullhistoryx(sorted_values(t));
%                 historyy(t) = fullhistoryy(sorted_values(t));
%                 historyz(t) = fullhistoryz(sorted_values(t));
%                 fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,historyx(t),historyy(t),historyz(t));
%             end
%             velocity = [fullhistoryx(sorted_values(t+1)) - fullhistoryx(sorted_values(t)), fullhistoryy(sorted_values(t+1)) - fullhistoryy(sorted_values(t)), fullhistoryz(sorted_values(t+1)) - fullhistoryz(sorted_values(t))];
%             fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n','velocity',velocity(1),velocity(2),velocity(3));
%         end
        %gt
        filename = sprintf('val_smooth/testgt%d-%d.txt', i,l);
        fileID = fopen(filename,'w');
        for t = 1:20
            if t == 1
                h = 'new';
            elseif t <= 10
                h = 'past';
            else
                h = 'future';
            end
            historyx(t) = fullhistoryx(sorted_values(t));
            historyy(t) = fullhistoryy(sorted_values(t));
            historyz(t) = fullhistoryz(sorted_values(t));
           fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,historyx(t),historyy(t),historyz(t));
        end
        %disrupt
        for d = 1:10
            filename = sprintf('val_smooth/testdisrupt%d-%d-%d.txt', i,l,d);
            fileID = fopen(filename,'w');
            for t = 1:10
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end
               newhistoryx = historyx(t);
               newhistoryy = historyy(t);
               newhistoryz = historyz(t);
               dt = 0.2 * (rand() - 0.5);
               dx = 0.1*historyx(t)*(rand()-0.5);
               dy = 0.1*historyy(t)*(rand()-0.5);
               dz = 0.1*historyz(t)*(rand()-0.5);
               newhistoryx = newhistoryx + dx;
               newhistoryy = newhistoryy + dy;
               newhistoryz = newhistoryz + dz;
               fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,newhistoryx,newhistoryy,newhistoryz);
            end
            for t = 11:20
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end            
               fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h, historyx(t),historyy(t),historyz(t));
            end
        end

        %disruption with variable musk
        for d = 1:10
            filename = sprintf('val_smooth/testmusk%d-%d-%d.txt', i,l,d);
            fileID = fopen(filename,'w');
            for t = 1:10
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end
               newhistoryx = historyx(t);
               newhistoryy = historyy(t);
               newhistoryz = historyz(t);
               dx = 0.1*historyx(t)*(rand()-0.5);
               dy = 0.1*historyy(t)*(rand()-0.5);
               dz = 0.1*historyz(t)*(rand()-0.5);
               newhistoryx = newhistoryx + dx;
               if rand()< 0.1
                   newhistoryx = newhistoryx + 10*dx;
               end
               newhistoryy = newhistoryy + dy;
               if rand()< 0.1
                   newhistoryy = newhistoryy + 10*dy;
               end
               newhistoryz = newhistoryz + dz;
               if rand()< 0.1
                   newhistoryz = newhistoryz + 10*dz;
               end

               fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,newhistoryx,newhistoryy,newhistoryz);
            end
            for t = 11:20
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end
               fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,historyx(t),historyy(t),historyz(t));
            end
        end
        %disruption with point missed
        for d = 1:10
            filename = sprintf('val_smooth/testpointmusk%d-%d-%d.txt', i,l,d);
            fileID = fopen(filename,'w');
            for t = 1:10
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end
               newhistoryx = historyx(t);
               newhistoryy = historyy(t);
               newhistoryz = historyz(t);
               dx = 0.1*historyx(t)*(rand()-0.5);
               dy = 0.1*historyy(t)*(rand()-0.5);
               dz = 0.1*historyz(t)*(rand()-0.5);
               newhistoryx = newhistoryx + dx;
               if rand()< 0.1
                   newhistoryx = newhistoryx + 10*dx;
               end
               newhistoryy = newhistoryy + dy;
               if rand()< 0.1
                   newhistoryy = newhistoryy + 10*dy;
               end
               newhistoryz = newhistoryz + dz;
               if rand()< 0.1
                   newhistoryz = newhistoryz + 10*dz;
               end
               if rand()> 0.15
                    fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h, newhistoryx,newhistoryy,newhistoryz);
               else
                    fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h,newhistoryx+10*dx,newhistoryy+10*dy,newhistoryz+10*dz);
               end
            end
            for t = 11:20
                if t == 1
                    h = 'new';
                elseif t <= 10
                    h = 'past';
                else
                    h = 'future';
                end
               fprintf(fileID,'%s\t%4.4f\t%4.4f\t%4.4f\n',h, historyx(t),historyy(t),historyz(t));
            end
        end
        figure(1)
        scatter3(historyx, historyy, historyz)
        plot3(historyx, historyy, historyz, 'o-')
        title('All Training Trajectory(Smooth Real Data)', 'FontSize', 14)
        xlabel('x', 'FontSize', 14)
        ylabel('y', 'FontSize', 14)
        zlabel('z', 'FontSize', 14)
        hold on
        xlim([-1.5 1.5])
        ylim([-1.5 1.5])
        zlim([-0.1 2])
    end

end
% str = sprintf('All Training Trajectory(Smooth Real Data).png');
% print(gcf,str,'-dpng','-r900'); 
fclose(fileID);

