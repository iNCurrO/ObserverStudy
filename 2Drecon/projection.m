function proj2d = projection(data2d,param, iview)

angle_rad = param.deg(iview)/360*2*pi;
proj2d = (zeros(param.nu, 1,'single'));

uu = param.us;
[xx,yy] = meshgrid(param.xs,param.ys);

if param.gpu == 1
    data2d = gpuArray(single(data2d));
    rx = gpuArray(((xx.*cos(angle_rad) - yy.*sin(angle_rad)) - xx(1,1))/param.dx + 1);
    ry = gpuArray(((xx.*sin(angle_rad) + yy.*cos(angle_rad)) - yy(1,1))/param.dy + 1);
else
    rx = (((xx.*cos(angle_rad) - yy.*sin(angle_rad)) - xx(1,1))/param.dx + 1);
    ry = (((xx.*sin(angle_rad) + yy.*cos(angle_rad)) - yy(1,1))/param.dy + 1);
end

% for iz = 1:param.nz   
data2d = interp2(data2d,rx,ry, param.interptype);
    

data2d(isnan(data2d))=0;


xx = param.xs;


for iy = 1:param.ny
    
    Ratio = (param.ys(iy)+param.DSO)/(param.DSD);
    
    pu = uu*Ratio; 
    
    pu = (pu - xx(1))/(param.dx)+1; 
    
%     if param.gpu == 1
%         tmp = gather(interp(gpuArray(single(data2d(:,iy))),gpuArray(single(pu)),param.interptype));
%     else
        tmp = (interp1((single(data2d(:,iy))),(single(pu)),param.interptype));
%     end
%     tmp = data2d(:,iy);
    
    tmp(isnan(tmp))=0;
    
    proj2d = proj2d + tmp';
end

dist = sqrt((param.DSD)^2 + uu.^2 )./(param.DSD)*param.dy;



proj2d = proj2d .* dist';





