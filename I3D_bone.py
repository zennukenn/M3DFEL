
from pytorchvideo.models.hub import i3d_r50
from utils import *

model_state_dict = torch.load('resnet50_3d.pth')
class M3DFEL_I3D(nn.Module):
    """The proposed M3DFEL framework

    Args:
        args
    """

    def __init__(self, args):
        super(M3DFEL_I3D, self).__init__()

        self.args = args
        self.device = torch.device(
            'cuda:%d' % args.gpu_ids[0] if args.gpu_ids else 'cpu')
        self.bag_size = self.args.num_frames // self.args.instance_length
        self.instance_length = self.args.instance_length

        # backbone networks
        self.features = i3d_r50(pretrained=False)
        # 将本地参数加载到模型中
        try:
            self.features.load_state_dict(model_state_dict, strict=True)
            print("模型参数加载成功")
        except RuntimeError as e:
            print(f"模型参数加载失败: {e}")

        self.features.blocks[-1] = torch.nn.Identity()  # 移除分类层
        self.features.blocks[-2] =torch.nn.AdaptiveAvgPool3d(output_size=(1, 1, 1))  # 修改池化层加一层全连接层
        # Load model parameters if available


        self.lstm = nn.LSTM(input_size=1024, hidden_size=512,  # Adjust input size according to i3d_r50 output
                            num_layers=2, batch_first=True, bidirectional=True)

        # multi head self attention
        self.heads = 8
        self.dim_head = 1024 // self.heads
        self.scale = self.dim_head ** -0.5
        self.attend = nn.Softmax(dim=-1)
        self.to_qkv = nn.Linear(
            1024, (self.dim_head * self.heads) * 3, bias=False)

        self.norm = DMIN(num_features=1024)
        self.pwconv = nn.Conv1d(self.bag_size, 1, 3, 1, 1)

        # classifier
        self.fc = nn.Linear(1024, self.args.num_classes)
        self.Softmax = nn.Softmax(dim=-1)

    def MIL(self, x):
        """The Multi Instance Learning Agregation of instances

        Inputs:
            x: [batch, bag_size, 2048]
        """
        self.lstm.flatten_parameters()
        x, _ = self.lstm(x)

        ori_x = x

        # MHSA
        qkv = self.to_qkv(x).chunk(3, dim=-1)
        q, k, v = map(lambda t: rearrange(
            t, 'b n (h d) -> b h n d', h=self.heads), qkv)
        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale
        attn = self.attend(dots)
        x = torch.matmul(attn, v)
        x = rearrange(x, 'b h n d -> b n (h d)')
        x = self.norm(x)
        x = torch.sigmoid(x)

        x = ori_x * x

        return x

    def forward(self, x):
        x = rearrange(x, 'b (t1 t2) c h w -> (b t1) c t2 h w',
                      t1=self.bag_size, t2=self.instance_length)

        x = self.features(x).squeeze()

        x = rearrange(x, '(b t) c -> b t c', t=self.bag_size)

        x = self.MIL(x)

        x = self.pwconv(x).squeeze()

        out = self.fc(x)

        return out



class Args:
    num_frames = 16  # Total number of frames
    instance_length = 4  # Frames per instance
    num_classes = 4  # Number of output classes
    gpu_ids = [0]  # List of GPU IDs, adjust according to your hardware

args = Args()

# Initialize the model
model = M3DFEL_I3D(args)
model.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move model to GPU if available
print(torch.cuda.is_available())

# Generate a random input tensor
# Assuming the input needs [batch_size, sequence_length, channels, height, width]
batch_size = 1
sequence_length = args.num_frames  # Total frames
channels = 3  # RGB channels
height = 112  # Height in pixels
width = 112  # Width in pixels
random_input = torch.randn(batch_size, sequence_length, channels, height, width)
random_input = random_input.to('cuda' if torch.cuda.is_available() else 'cpu')  # Move tensor to GPU if available

# Forward pass through the model
try:
    output = model(random_input)
    print("Model output:", output)
except Exception as e:
    print("An error occurred during model execution:", str(e))