clear all;
simkai=imread('yisanshu.png');
[fid,message]=fopen('yisanshu.txt', 'r');
graykai=rgb2gray(simkai);
for i=1:size(graykai,1)
    if mod(i,80)==0
        graykai(i,:)=0;
    end
end
h_kai=zeros(80,73*10);
for i=1:10
    h_kai(1:80,73*(i-1)+1:73*i)=graykai(80*i-79:80*i,1:73);
end
figure(1)
imshow(h_kai);
figure(2)
imshow(graykai);
wan=graykai(561:640,:);
figure(3)
imshow(wan);
corners = detectFASTFeatures(wan,'MinContrast',0.1);
J = insertMarker(wan,corners,'circle');
figure(4)
imshow(J);