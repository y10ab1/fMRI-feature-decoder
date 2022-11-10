from torch import nn, randn
from torchsummary import summary



class AAEEncoder(nn.Module):

    def __init__(self, num_classes):
        super().__init__()
        
        self.conv1 = nn.Sequential(
            nn.Conv3d(in_channels=1, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Conv3d(in_channels=64, out_channels=64, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(64),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=2),
            
        )
        #self.conv1 = nn.DataParallel(self.conv1)

        self.conv2 = nn.Sequential(
            nn.Conv3d(in_channels=64, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Conv3d(in_channels=128, out_channels=128, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(128),
            nn.ReLU(),

            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=2),
        )
        #self.conv2 = nn.DataParallel(self.conv2)

        self.conv3 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),#
            nn.Dropout(p=0.3),
            
            nn.Conv3d(in_channels=256, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            nn.Conv3d(in_channels=384, out_channels=384, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(384),
            nn.ReLU(),#
            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=2),
        )
        #self.conv3 = nn.DataParallel(self.conv3)

        self.conv4 = nn.Sequential(
            nn.Conv3d(in_channels=384, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Conv3d(in_channels=512, out_channels=512, kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm3d(512),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            
            #nn.MaxPool3d(kernel_size=(1,2)),
            nn.MaxPool3d(kernel_size=(1)),
        )
        #self.conv4 = nn.DataParallel(self.conv4)

        self.conv5 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, stride=1, padding=1),
            
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),
            nn.MaxPool3d(kernel_size=(2)),
        )
        #self.conv5 = nn.DataParallel(self.conv5)

        self.conv6 = nn.Sequential(
            nn.Conv3d(in_channels=128, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3),

            nn.Conv3d(in_channels=256, out_channels=256, kernel_size=3, padding=1),
            nn.BatchNorm3d(256),
            nn.ReLU(),
            nn.Dropout(p=0.3)
        )
        #self.conv6 = nn.DataParallel(self.conv6)
###########################################################################
# For classify
        self.conv7 = nn.Sequential(
            nn.Conv3d(in_channels=256, out_channels=num_classes, kernel_size=1, padding=1),
            nn.BatchNorm3d(num_classes),
            nn.ReLU(),
            nn.AdaptiveAvgPool3d(1)
        )
        #self.conv7= nn.DataParallel(self.conv7)

        self.flatten = nn.Flatten()
        self.softmax = nn.Softmax(dim=1)
        self.Hardtanh = nn.Hardtanh(min_val=0.0, max_val=1.0)
        self.linear = nn.Linear(1, 1)
###########################################################################

    def forward(self, input_data):
        x = self.conv1(input_data)
        x = self.conv2(x)
        x = self.conv6(x)
        x = self.conv7(x)
        #x = self.Hardtanh(x)
        x = self.linear(x)
        return x

if __name__ == "__main__":
    encoder = AAEEncoder(num_classes = 1).cuda()
    print(encoder)
    x = randn(1,1,10,10,10).cuda()
    y = encoder(x)
    print(y, y.shape)
    summary(encoder.cuda(), (1, 10, 10, 10))
    