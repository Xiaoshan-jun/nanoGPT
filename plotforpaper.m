
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
point = [];
point1 = [];
point2 = [];
point4 = [];
point4 = [];
point5 = [];
point6 = [];
point7 = [];
point8 = [];
point9 = [];
poingt10 = [];
for i = 1 : 1 %trajectory number
    if mod(i,4) == 0
    state = [400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4) == 1
    state = [-400 + 200*(rand()-0.5) , 400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4) == 2
    state = [400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    elseif mod(i,4)==3
    state = [-400 + 200*(rand()-0.5) , -400 + 200*(rand()-0.5), 140 + 15*(rand()-0.5)];
    end
    destination = [0, 0, 0];
    first = 1;        

    for t = 1 : 20
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
        
        historyx(t) = state(1);
        historyy(t) = state(2);
        historyz(t) = state(3);
    end
    %disrupt
    for d = 1:100
        for t = 1:20
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
           point = [point;[newhistoryx, newhistoryy, newhistoryz]];
        end

    end

% 
%     figure(1)
%     scatter3(historyx, historyy, historyz)
%     plot3(historyx, historyy, historyz, 'o-')
%     title('linear landing', 'FontSize', 14)
%     xlabel('x', 'FontSize', 14)
%     ylabel('y', 'FontSize', 14)
%     zlabel('z', 'FontSize', 14)
%     hold on
end
%str = sprintf('linear%d.png', i);
%print(gcf,str,'-dpng','-r900'); 

[F, V] = boundary(point, 0);
vertices = V;
faces = F;
trisurf(faces, point(:,1), point(:,2), point(:,3));
hold on
% Compute the convex hull using the convhulln function
point1 = point(1:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point1);
vertices = point1;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
title('linear landing', 'FontSize', 14)
xlabel('x', 'FontSize', 14)
ylabel('y', 'FontSize', 14)
zlabel('z', 'FontSize', 14)
point2 = point(2:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point2);
vertices = point2;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
point3 = point(3:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point3);
vertices = point3;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
point4 = point(4:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point4);
vertices = point4;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
point5 = point(5:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point5);
vertices = point5;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
point6 = point(6:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point6);
vertices = point6;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));

point7 = point(7:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point7);
vertices = point7;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));


point8 = point(8:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point8);
vertices = point8;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));


point9 = point(9:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point9);
vertices = point9;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));

point10 = point(10:20:end,:);
% Generate a mesh using the vertices and faces of the convex hull
K = convhulln(point10);
vertices = point10;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));
%%
% Generate some example 3D points
x = randn(100,1);
y = randn(100,1);
z = randn(100,1);

% Calculate the convex hull of the points
K = boundary(x,y,z);

% Calculate the confidence level for each face
conf = rand(size(K,1),1); % Replace this with your own confidence levels

% Color the faces based on their confidence levels
trisurf(K, x, y, z, 'FaceColor', 'interp', 'FaceVertexCData', conf);

% Add a colorbar to show the mapping between confidence level and color
colorbar;
%% 
[F, V] = boundary(point, 0);
vertices = V;
faces = F;
trisurf(faces, point(:,1), point(:,2), point(:,3));
hold on
