img = zeros(129,129,129);
img(65,65,65) = 1;

for pixel_num = 1:200
    for direction = ['L', 'R', 'F', 'B', 'U', 'D']
        img = growthlesion(img, direction);
    end
end

function [resultImg] = growthlesion(targetImg, direction)
datasize = size(targetImg,1 );
switch direction
    case 'L'
        trans = max(targetImg, [], 1);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(:,targetIndex1, targetIndex2);
        loopFlag = 0;
        idx = 1;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx + 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg(idx - 1, targetIndex1, targetIndex2) = 1;
    case 'R'
        trans = max(targetImg, [], 1);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(:,targetIndex1, targetIndex2);
        loopFlag = 0;
        idx = datasize;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx - 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg(idx + 1, targetIndex1, targetIndex2) = 1;
    case 'F'
        trans = max(targetImg, [], 2);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(targetIndex1, :, targetIndex2);
        loopFlag = 0;
        idx = 1;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx + 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg( targetIndex1, idx - 1, targetIndex2) = 1;
    case 'B'
        trans = max(targetImg, [], 2);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(targetIndex1, :, targetIndex2);
        loopFlag = 0;
        idx = datasize;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx - 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg( targetIndex1, idx + 1, targetIndex2) = 1;
    case 'U'
        trans = max(targetImg, [], 3);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(targetIndex1, targetIndex2, :);
        loopFlag = 0;
        idx = 1;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx + 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg( targetIndex1,targetIndex2,  idx - 1) = 1;
    case 'D'
        trans = max(targetImg, [], 3);
        axisIndex = find(trans == 1);
        targetIndex = datasample(axisIndex, 1);
        targetIndex1 = mod(targetIndex, datasize);
        targetIndex2 = (targetIndex - targetIndex1)/datasize + 1;
        pixelList = targetImg(targetIndex1, targetIndex2, :);
        loopFlag = 0;
        idx = datasize;
        while loopFlag == 0
            if pixelList(idx) == 0
                idx = idx - 1;
            else
                loopFlag = 1;
            end
        end
        resultImg = targetImg;
        resultImg( targetIndex1,targetIndex2,  idx + 1) = 1;
end
end
