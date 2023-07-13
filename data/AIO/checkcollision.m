function c = checkcollision(x,y,z, obstacle)
    % Add two numbers
    c = 0;
    for i = 1: 10
        obstaclex = obstacle(i,1);
        obstacley = obstacle(i,2);
        obstaclez = obstacle(i,3);
        l = obstacle(i,4);
        w = obstacle(i,5);
        h = obstacle(i,6);
        if x > obstaclex && x < obstaclex + l && y > obstacley && y < obstacley + w && z > obstaclez && z <obstaclez + h
            c = 1;
            break
        end
end