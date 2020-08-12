%% Initialize
n_points = 39;
% azimuth of the tilt plane normal
psi  = 55+180;
% angle of circle
phi = linspace(0,2*pi,n_points)';
% principle angle (radians)
beta = 0.9; %(25 + rand(1) * 30) * pi / 180;
% circle of points at base of cone
p0 = [tan(beta)*cos(phi),tan(beta)*sin(phi),ones(size(phi))];
px = p0(:, 1);   py = p0(:, 2);   pz = p0(:, 3);
% optical axis
z = [zeros(size(phi)),  zeros(size(phi)),  ones(size(phi))];
% iterate plane normal's polar angle in degrees [ 0, ..., 360].
for theta = 0:6:360 
    % plane normal
    nx = sind(theta)*cosd(psi);
    ny = sind(theta)*sind(psi);
    nz = -cosd(theta);
    % rotation axis
    aXs = [linspace(-2, 2, n_points)', ...
           -(nx/ny)*linspace(-2, 2, n_points)', ...
           ones(size(phi))];
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Forward map 
    % lambda scale factor [equation 1] 
    l1 = nz ./ (nx*px + ny*py + nz*pz); 
    ll1 = [l1, l1, l1];  % stack as n x 3 array 
    % rodrigues rotation matrix [equation 2] 
    R = [1+(nx^2/(nz-1)),  nx*ny/(nz-1),  nx; 
            nx*ny/(nz-1), 1+ny^2/(nz-1),  ny; 
                     -nx,           -ny, -nz]; 
    % tilt shift map [equations 3 & 4] 
    p1 = (ll1.*p0 - z)*R' + z; 
    p1_x = (((nx^2+nz*(nz-1))*px + nx*ny*py)./(nx*px+ny*py+nz))/(nz-1); 
    p1_y = (((ny^2+nz*(nz-1))*py + nx*ny*px)./(nx*px+ny*py+nz))/(nz-1); 
    %% Backward map
    % lambda scale factor [equation 6]
    l6 = 1 ./ (nx*p1_x + ny*p1_y + 1);
    ll6 = [l6, l6, l6];  % stack as n x 3 array
    % tilt corrected map [equations 5 & 7]
    p0_b = ll6.*((p1 - z)*R + z); 
    p0_x = (((nx^2+nz-1)*p1_x + nx*ny*p1_y)./(nx*p1_x+ny*p1_y+1))/(nz-1);
    p0_y = (((ny^2+nz-1)*p1_y + nx*ny*p1_x)./(nx*p1_x+ny*p1_y+1))/(nz-1);
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Tilted plane
    [plane_x,plane_y] = meshgrid(linspace(-3,3,101),linspace(-3,3,101));
    plane_flat_z = ones(size(plane_x));
    plane_tilt_z = - nx / nz * plane_x - ny / nz * plane_y + ones(size(plane_x));
    % cone plane intersection 3d
    origin = zeros(n_points, 1);
    intersect = ll1.*p0;
    %%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%%
    %% Plotting
    h=figure(33);clf;
    subplot(1,2,1)
    hold on;grid on;
    surf(plane_x,plane_y,plane_flat_z,'FaceAlpha',0.5,'FaceColor','interp');
    surf(plane_x,plane_y,plane_tilt_z,'FaceAlpha',0.5,'FaceColor','interp');
    plot3(aXs(:,1), aXs(:,2), aXs(:,3), 'c--', 'LineWidth', 3)
    plot3(p0(:,1), p0(:,2), p0(:,3),'color',[0.725 0.725 0.725],'LineWidth', 3)
    plot3(intersect(:,1), intersect(:,2), intersect(:,3), 'k-', intersect(:,1), intersect(:,2), intersect(:,3), 'r*', 'LineWidth', 3)
    plot3([intersect(:,1), 2*p0(:,1)]', [intersect(:,2), 2*p0(:,2)]', [intersect(:,3), 2*p0(:,3)]','color',[0.725 0.725 0.725],'LineWidth',1)
    plot3(2*p0(:,1), 2*p0(:,2), 2*p0(:,3),':','color',[0.725 0.725 0.725],'LineWidth', 1)
    plot3(p1(:,1), p1(:,2), p1(:,3), 'g-', p1(:,1), p1(:,2), p1(:,3), 'r.', 'LineWidth', 3)
    xlabel('x')
    ylabel('y')
    zlabel('z')
    % annotation('textbox', [0.385, 0.705, 0.0725, 0.025], 'String', round(theta) + " degrees tilt")
    plot3([origin, intersect(:,1)]', [origin, intersect(:,2)]', [origin, intersect(:,3)]','color',[0.725 0.725 0.725],'LineWidth',3)
    colormap winter
    shading interp
    set(gca,'Zdir','reverse')
    view([10 10])
    axis([-3 3 -3 3 0 2])
    axis square
    subplot(1,2,2)
    hold on;grid off;
    plot(p0(:,1),p0(:,2),'*',p0(:,1),p0(:,2),'-','color',[0.725 0.725 0.725],'LineWidth',2)
    plot(p1(:,1),p1(:,2),'*',p1(:,1),p1(:,2),'-','color',[0.525 0.925 0.525],'LineWidth',2)
    plot(px, py,'c.', p1_x, p1_y, 'r.', p0_b(:, 1), p0_b(:, 2), 'r^', p0_x, p0_y, 'ko','LineWidth',1)
    plot(aXs(:,1), aXs(:,2), 'c--','LineWidth',1)
    axis([-5 5 -5 5])
    axis square
    xlabel('x')
    ylabel('y')
    % save png for gif
    % saveas(h,sprintf('makegif/conic%d.png',theta+100));
end
