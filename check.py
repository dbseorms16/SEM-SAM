import torch


encoder = torch.load('./pre_trained/sam_vit_h_4b8939.pth')
decoder = torch.load('./pre_trained/sam_vit_h_maskdecoder.pth')

for k in encoder.keys():
    print(k)

# print(encoder)
# print(decoder)