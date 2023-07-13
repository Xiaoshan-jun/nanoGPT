%% 
%straight line
%fileID = fopen('train.txt','w');
fileID = fopen('val.txt','w');
for kk = 1:1000
    %Linear
    fprintf(fileID,'\nnew');
    %obstacle
    obstacle = zeros(10,6);
    for i = 1: 10
        obstaclex = round(600*rand());
        obstacley = round(600*rand());
        obstaclez = round(600*rand());
        l = round(40*rand() + 20);
        w = round(40*rand() + 20);
        h = round(40*rand() + 20);
        obstacle(i,:) = [obstaclex,obstacley,obstaclez, l, w, h];
        fprintf(fileID,'\no:%i\t%i\t%i\t%i\t%i\t%i', obstaclex,obstacley,obstaclez, l, w, h);
    end
    %max velocity
    mvx = round(50*(rand()+0.1) + 10); %max horizontal speed
    mvy = round(50*(rand()+0.1) + 10); %max horizontal speed
    mvz = round(25*(rand()+0.1) + 5);  %max descend speed
    fprintf(fileID,'\nmv:%i\t%i\t%i\t', mvx, mvy, mvz);
    %origin
    originx = round(600 * rand());
    originy = round(600 * rand());
    originz = round(600 * rand());
    c = checkcollision(originx,originy,originz, obstacle);
    while c
        originx = round(600 * rand());
        originy = round(600 * rand());
        originz = round(600 * rand());
        c = checkcollision(originx,originy,originz, obstacle);
    end
    fprintf(fileID,'\ns:%i\t%i\t%i\t',originx , originy, originz);
    %destination
    destinaionx = round(600 * rand());
    destinaiony = round(600 * rand());
    destinaionz = round(600 * rand());
    c = checkcollision(destinaionx,destinaiony,destinaionz, obstacle);
    while c
        destinaionx = 600 * rand();
        destinaiony = 600 * rand();
        destinaionz = 600 * rand();
        c = checkcollision(destinaionx,destinaiony,destinaionz, obstacle);
    end
    fprintf(fileID,'\nd:%i\t%i\t%i\t',destinaionx , destinaiony, destinaionz);
    departtime = round(900*rand());
    %trajectory
    cx = originx;
    cy = originy;
    cz = originz;
    xd = destinaionx - cx;
    yd = destinaiony - cy;
    zd = destinaionz - cz;
    t = departtime;
    historyx = [];
    historyy = [];
    historyz = [];
    historyv = [];
    while abs(xd) > 10 || abs(yd) > 10 || abs(zd) >10
        if xd > 0
            xv = round(mvx*rand());
        else
            xv = -round(mvx*rand());
        end
        if yd > 0
            yv = round(mvy*rand());
        else
            yv = -round(mvy*rand());
        end
        if zd > 0
            zv = round(mvz*rand());
        else
            zv = -round(mvz*rand());
        end
        c = checkcollision(cx + xv, cy + yv, cz + zv, obstacle);
        while c
            if xd > 0
                xv = round(min(mvx, abs(xd))*rand());
            else
                xv = -round(min(mvx, abs(xd))*rand());
            end
            if yd > 0
                yv = round(min(mvy, abs(yd))*rand());
            else
                yv = -round(min(mvy, abs(yd))*rand());
            end
            if zd > 0
                zv = round(min(mvx, abs(zd))*rand());
            else
                zv = -round(min(mvx, abs(zd))*rand());
            end
        c = checkcollision(cx + xv, cy + yv, cz + zv, obstacle);
        end
        cx = cx + xv;
        cy = cy + yv;
        cz = cz + zv;
        t = t + 1;
        historyx = [historyx, cx];
        historyy = [historyy, cy];
        historyz = [historyz, cz];
        historyv = [historyv, sqrt(xv^2 + yv^2 + zv^2)];
        if (t - departtime < 10 && rem(t, 2) == 0) || (t - departtime < 30 && rem(t, 3) == 0) || (t - departtime < 100 && rem(t, 5) == 0)  
            fprintf(fileID,'\np:%i\t%i\t%i\t%i',t, cx,cy,cz);
        end
        if t - departtime > 100
            break
        end
        xd = destinaionx - cx;
        yd = destinaiony - cy;
        zd = destinaionz - cz;
    end
    at = t+1;
    fprintf(fileID,'\ndt:%i\tat:%i', departtime,at);
    meanv = mean(historyv);
    variancev = sqrt(var(historyv));
    fprintf(fileID,'\nav:%.2f\tvv:%.2f', meanv,variancev);         
%     figure(1)
%     scatter3(historyx, historyy, historyz)
%     plot3(historyx, historyy, historyz, 'o-')
%     title('linear landing', 'FontSize', 14)
%     xlabel('x', 'FontSize', 14)
%     ylabel('y', 'FontSize', 14)
%     zlabel('z', 'FontSize', 14)
%     hold on
end