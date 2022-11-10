from torch import nn, randn
from torchsummary import summary



class AAEEncoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        self.lazylinear = nn.LazyLinear(out_features=128)
        self.linear = nn.Linear(128, 128)
        self.linear_output = nn.Linear(128, 1)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0)
###########################################################################

    def forward(self, input_data):
        x = self.lazylinear(input_data)
        x = self.linear(x)

        x = self.linear_output(x)
        x = self.Hardtanh(x)
        return x

if __name__ == "__main__":
    encoder = AAEEncoder(num_classes = 1).cuda()
    #print(encoder)
    x = randn(1,1,100).cuda()
    y = encoder(x)
    print(y, y.shape)
    summary(encoder.cuda(), (1, 100))
    