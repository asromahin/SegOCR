import torch
import timm


class CRNN(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_classes,
            in_channels=3,
            rnn_size=64,
            encoder_name='efficientnet-b0',
            pretrained=True,
    ):
        super(CRNN, self).__init__()
        self.input_size = input_size
        self.in_channels = in_channels
        self.rnn_size = rnn_size
        self.output_classes = output_classes

        self.backbone = timm.create_model(
            model_name=encoder_name,
            in_channels=in_channels,
            pretrained=pretrained,
            features_only=True,
        )
        self.features_channels = self._get_backbon_out_channels()

        self.activation = torch.nn.LogSoftmax(dim=2)

    def _get_backbon_out_channels(self):
        placeholder = torch.zeros((1, self.in_channels, self.input_size[1], self.input_size[0]), dtype=torch.float, device=self.backbone.device)
        last_output = self.backbone(placeholder)[-1]
        return last_output.shape[1]

    def forward(self, x):
        feature = self.backbone(x)[-1]
