from torch import nn, randn, hub
from torchsummary import summary



class GoogleNet_transfer(nn.Module):

    def __init__(self, out_features=1, use_pretrained=True):
        
        super().__init__()
        
        model = hub.load('pytorch/vision:v0.10.0', 'googlenet', pretrained=use_pretrained)

        for param in model.parameters():
            param.requires_grad = False
        in_features = model.fc.in_features

        self.backbone = model
        self.backbone.fc = nn.Linear(in_features=in_features, out_features=out_features)
        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.Relu = nn.ReLU()


    def forward(self, x):
        x = self.backbone(x)
        #x = self.Hardtanh(x)
        x = self.Relu(x)
        return x

if __name__ == "__main__":
    encoder = AAEEncoder(num_classes = 1).cuda()
    #print(encoder)
    x = randn(1,1,100).cuda()
    y = encoder(x)
    print(y, y.shape)
    summary(encoder.cuda(), (1, 100))
    