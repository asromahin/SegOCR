import segmentation_models_pytorch as smp
import torch


class ModelTypeEnum:
    unet = 'unet'
    unetplusplus = 'unetplusplus'
    linknet = 'linknet'
    deeplabv3 = 'deeplabv3'
    fpn = 'fpn'
    pan = 'pan'
    pspnet = 'pspnet'


class SegOCR(torch.nn.Module):
    def __init__(
            self,
            input_size,
            output_classes,
            in_channels=3,
            rnn_size=64,
            encoder_name='resnet34',
            model_type=ModelTypeEnum.unet,
            pretrained=True,
    ):
        super(SegOCR, self).__init__()
        self.input_size = input_size
        self.rnn_size = rnn_size
        self.output_classes = output_classes

        params = {
            'encoder_name': encoder_name,
            'in_channels': in_channels,
            'classes': output_classes,
        }
        if not pretrained:
            params['encoder_weights'] = None
        if model_type == ModelTypeEnum.unet:
            params['decoder_attention_type'] = 'scse'
            self.seg_model = smp.Unet(**params)
        elif model_type == ModelTypeEnum.linknet:
            self.seg_model = smp.Linknet(**params)

        convert_size = (self.input_size[0], self.input_size[1]//rnn_size)
        # Output size after this conv - (batch_size, output_classes, 1, rnn_size)
        self.to_rnn_size = torch.nn.Conv2d(
            self.output_classes,
            self.output_classes,
            kernel_size=convert_size,
            stride=convert_size,
        )
        self.activation = torch.nn.LogSoftmax(dim=2)
        self.seg_activation = torch.nn.LogSoftmax(dim=1)

    def forward(self, x, return_seg=False):
        seg_out = self.seg_model(x)
        logits = self.to_rnn_size(seg_out)
        logits = logits.squeeze(dim=2)
        logits = logits.permute(2, 0, 1)
        logits = self.activation(logits)
        seg_out = self.seg_activation(seg_out)
        if return_seg:
            return logits, seg_out
        else:
            return logits
