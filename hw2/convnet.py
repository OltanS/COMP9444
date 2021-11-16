class ConvNet(nn.Module):
    def __init__(self):
        super().__init__()

        self.convolutional_layer = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=6, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=0),
            nn.Conv2d(in_channels=6, out_channels=16, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=0),
            nn.Conv2d(in_channels=16, out_channels=120, kernel_size=(5,5)),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=(2,2),stride=(2,2),padding=0),
            nn.Conv2d(in_channels=120, out_channels=120, kernel_size=(5,5)),
            nn.ReLU()          
        )

        self.linear_layer = nn.Sequential(
            nn.Dropout(),
            nn.Linear(in_features=2*2*120, out_features=84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=84, out_features=84),
            nn.ReLU(),
            nn.Dropout(),
            nn.Linear(in_features=84, out_features=8),
            nn.ReLU()
        )
        
    def forward(self, input):
        input = self.convolutional_layer(input)
        # print(f"input shape: {input.shape}")
        input = input.view(-1,2*2*120)
        output = self.linear_layer(input)  
        return output
