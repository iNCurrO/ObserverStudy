function vol = backprojection(proj,param,iview)

angle_rad = param.deg(iview)/360*2*pi;
vol = zeros(param.nx,param.ny,'single');

[xx,yy] = meshgrid(param.xs,param.ys);

rx = xx.*cos(angle_rad-pi/2) + yy.*sin(angle_rad-pi/2);
ry = -xx.*sin(angle_rad-pi/2) + yy.*cos(angle_rad-pi/2);

pu = single(((rx.*(param.DSD)./(ry + param.DSO))+param.us(1))/(-param.du) + 1);
Ratio = (single(param.DSO.^2./(param.DSO+ry).^2));
if param.gpu == 1
    pu = gpuArray(single(pu));
    proj = gpuArray(single(proj));
    Ratio = gpuArray(Ratio);
end

if param.gpu == 1
    vol(:,:) = gather(Ratio.*interp(proj',pu,param.interptype));
else
    vol(:,:) = (Ratio.*interp(proj',pu,param.interptype));
end

vol(isnan(vol))=0;

return

















