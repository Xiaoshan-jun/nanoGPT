%%
%straight line
%fileID = fopen('datasets/linear/train/train.txt','w');
%fileID = fopen('datasets/linear/val/val.txt','w');
%fileID = fopen('datasets/linear/vis/vis.txt','w');
%fileID = fopen('train.txt','w');
%fileID = fopen('val.txt','w');
% fileID = fopen('testgt.txt','w');
% fileID = fopen('testdisrupt.txt','w');
% fileID = fopen('testmusk.txt','w');
% fileID = fopen('testpointmusk.txt','w');
mvx = 25; %max horizontal speed
mvy = 25; %max horizontal speed
mvz = 9;  %max descend speed

for i = 1 : 12 %trajectory number
    if mod(i,4) == 0
    state = [400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4) == 1
    state = [-400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4) == 2
    state = [400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4)==3
    state = [-400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    end
    first = 1;        
    filename = sprintf('val/testgt%d.txt', i);
    fileID = fopen(filename,'w');
    destination = [0, 0, state(3)];
    for t = 1 : 9
        vb = (destination - state)/(11 - t);
        if mod(i,4) == 0
        xv = max(-mvx , vb(1) + 15 * (rand() - 0.5));
        yv = max(-mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4) == 1
        xv = min(mvx , vb(1) + 15 * (rand() - 0.5));
        yv = max(-mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4) == 2
        xv = max(-mvx , vb(1) + 15 * (rand() - 0.5));
        yv = min(mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4)==3
        xv = min(mvx , vb(1) + 15 * (rand() - 0.5));
        yv = min(mvy, vb(2) + 15*(rand() - 0.5));
        end

        zv = max(-mvz, vb(3) + 15*(rand() - 0.5));

        state(1) = state(1) + xv;
        state(2) = state(2) + yv;
        state(3) = max(state(3) + zv,0); % bound z above zero
        if first == 1
            h = 'new';
            first = 0;
        elseif t <= 10
            h = 'past';
        else
            h = 'future';
        end
        fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,state(1),state(2),state(3));
        
        historyx(t) = state(1);
        historyy(t) = state(2);
        historyz(t) = state(3);
    end
    destination = [0, 0, 0];
    for t = 10 : 20
        vb = (destination - state)/(21 - t);
        if mod(i,4) == 0
        xv = max(-mvx , vb(1) + 15 * (rand() - 0.5));
        yv = max(-mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4) == 1
        xv = min(mvx , vb(1) + 15 * (rand() - 0.5));
        yv = max(-mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4) == 2
        xv = max(-mvx , vb(1) + 15 * (rand() - 0.5));
        yv = min(mvy, vb(2) + 15*(rand() - 0.5));
        elseif mod(i,4)==3
        xv = min(mvx , vb(1) + 15 * (rand() - 0.5));
        yv = min(mvy, vb(2) + 15*(rand() - 0.5));
        end

        zv = max(-mvz, vb(3) + 15*(rand() - 0.5));

        state(1) = state(1) + xv;
        state(2) = state(2) + yv;
        state(3) = max(state(3) + zv,0); % bound z above zero
        if first == 1
            h = 'new';
            first = 0;
        elseif t <= 10
            h = 'past';
        else
            h = 'future';
        end
        fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,state(1),state(2),state(3));
        
        historyx(t) = state(1);
        historyy(t) = state(2);
        historyz(t) = state(3);
    end
    %disrupt
    for d = 1:10
        filename = sprintf('val/testdisrupt%d.txt', 10*i + d);
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
           dx = 5*(rand()-0.5);
           dy = 5*(rand()-0.5);
           dz = 0.5*(rand()-0.5);
           newhistoryx = newhistoryx + dx;
           newhistoryy = newhistoryy + dy;
           newhistoryz = newhistoryz + dz;
           fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t+dt,newhistoryx,newhistoryy,newhistoryz);
        end
        for t = 11:20
            if t == 1
                h = 'new';
            elseif t <= 10
                h = 'past';
            else
                h = 'future';
            end
           fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,historyx(t),historyy(t),historyz(t));
        end
    end

    %disruption with variable musk
    for d = 1:10
        filename = sprintf('val/testmusk%d.txt', 10*i + d);
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
           dx = 5*(rand()-0.5);
           dy = 5*(rand()-0.5);
           dz = 0.5*(rand()-0.5);
           newhistoryx = newhistoryx + dx;
           if rand()< 0.1
               newhistoryx = 0;
           end
           newhistoryy = newhistoryy + dy;
           if rand()< 0.1
               newhistoryy = 0;
           end
           newhistoryz = newhistoryz + dz;
           if rand()< 0.1
               newhistoryz = 0;
           end
           
           fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t+dt,newhistoryx,newhistoryy,newhistoryz);
        end
        for t = 11:20
            if t == 1
                h = 'new';
            elseif t <= 10
                h = 'past';
            else
                h = 'future';
            end
           fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,historyx(t),historyy(t),historyz(t));
        end
    end
    %disruption with point missed
    for d = 1:10
        filename = sprintf('val/testpointmusk%d.txt', 10*i + d);
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
           dx = 5*(rand()-0.5);
           dy = 5*(rand()-0.5);
           dz = 0.5*(rand()-0.5);
           newhistoryx = newhistoryx + dx;
           if rand()< 0.1
               newhistoryx = 0;
           end
           newhistoryy = newhistoryy + dy;
           if rand()< 0.1
               newhistoryy = 0;
           end
           newhistoryz = newhistoryz + dz;
           if rand()< 0.1
               newhistoryz = 0;
           end
           if rand()> 0.15
            fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t+dt,newhistoryx,newhistoryy,newhistoryz);
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
           fprintf(fileID,'%s\t%2.1f\t%4.4f\t%4.4f\t%4.4f\n',h, t,historyx(t),historyy(t),historyz(t));
        end
    end
    figure(1)
    scatter3(historyx, historyy, historyz)
    plot3(historyx, historyy, historyz, 'o-')
    title('vertical landing', 'FontSize', 14)
    xlabel('x', 'FontSize', 14)
    ylabel('y', 'FontSize', 14)
    zlabel('z', 'FontSize', 14)
    hold on
end
%str = sprintf('linear%d.png', i);
%print(gcf,str,'-dpng','-r900'); 
fclose(fileID);
% points = [transpose(historyx), transpose(historyy), transpose(historyz)];
% Compute the convex hull using the convhulln function


% Generate a mesh using the vertices and faces of the convex hull
% vertices = points;
% faces = K;
% trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
