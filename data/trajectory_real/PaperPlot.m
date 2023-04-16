filename = sprintf('realdata2/flightdata%d.csv', 0);
T = readtable(filename);
T = table2array(T);
historyx = [];
historyy =[];
historyz =[];
figure(1)
i = 1;
for k = 1:size(T)
    r = rem( k , 100 );
    if r ~= 0
        continue
    else
        historyx(i) = T(k, 2);
        historyy(i) = T(k, 3);
        historyz(i) = T(k, 4);
         i = i + 1;
    end
end
hf = floor(length(historyx)/2);
hf = hf(1);
%scatter3(historyx, historyy, historyz)
plot3(historyx(1:hf), historyy(1:hf), historyz(1:hf), 'r-','DisplayName', 'Past Trajectory Ground Truth')
hold on
plot3(historyx(hf:length(historyx)), historyy(hf:length(historyx)), historyz(hf:length(historyx)), 'b-', 'DisplayName', 'Future Trajectory Ground Truth')
title("Future Trajectory", 'FontSize', 14)
xlabel('x', 'FontSize', 14)
ylabel('y', 'FontSize', 14)
zlabel('z', 'FontSize', 14)
grid on
xlim([-0.1 0.3])
ylim([-0 1.5])
zlim([-0.1 2])
historyx = [];
historyy =[];
historyz =[];
figure(1)
i = 1;
for k = 1:size(T)
    r = rem( k , 100 );
    if r ~= 0
        continue
    else
        historyx(i) = T(k, 2) + 0.1*(rand() - 0.5);
        historyy(i) = T(k, 3) + 0.1*(rand() - 0.5);
        historyz(i) = T(k, 4) + 0.1*(rand() - 0.5);
        i = i + 1;
    end
end
hf = floor(length(historyx)/2);
hf = hf(1);
%scatter3(historyx, historyy, historyz)
plot3(historyx(1:hf), historyy(1:hf), historyz(1:hf), 'or:','DisplayName', 'Past Trajectory Sensored')
% plot3(historyx(hf:length(historyx)), historyy(hf:length(historyx)), historyz(hf:length(historyx)), 'ob:', 'DisplayName', 'Future Trajectory Sensored')
legend('location', 'Best');
print(gcf,'Trajectory predicting2','-dpng','-r900');
%%

%scatter3(historyx, historyy, historyz)
% plot3(historyx(hf:length(historyx)), historyy(hf:length(historyx)), historyz(hf:length(historyx)), 'ob:', 'DisplayName', 'Future Trajectory Sensored')
%legend('location', 'Best');
%print(gcf,'Trajectory predicting2','-dpng','-r900');
% Set up a set of 3D points
points = randn(100,3);
points = [transpose(historyx), transpose(historyy), transpose(historyz)];
% Compute the convex hull using the convhulln function
K = convhulln(points);

% Generate a mesh using the vertices and faces of the convex hull
vertices = points;
faces = K;
trisurf(faces, vertices(:,1), vertices(:,2), vertices(:,3));

