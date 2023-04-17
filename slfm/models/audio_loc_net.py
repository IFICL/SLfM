import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio
from tqdm import tqdm

import sys
sys.path.append('..')
from config import init_args, params
from utils import utils, torch_utils


class CondAudioEncoder(nn.Module):
    def __init__(self, args, pr):
        super(CondAudioEncoder, self).__init__()
        self.pr = pr
        self.args = args
        self.net = self.construct_audio_net(args, pr)
        if args.add_geometric:
            out_dim = 1
            self.pred_head = nn.Sequential(
                nn.Linear(pr.audio_feature_dim, int(pr.audio_feature_dim // 2)),
                nn.ReLU(True),
                nn.Linear(int(pr.audio_feature_dim // 2), out_dim)
            )
            for m in self.pred_head.modules():
                if isinstance(m, nn.Linear):
                    nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)

    def forward(self, audio, return_angle=False, augment=False):
        # import pdb; pdb.set_trace()
        ''' 
            audio: (N, C, L)
        '''
        # Re-cut the conditional audio clip to meet the requirement
        audio = audio[..., :int(self.pr.cond_clip_length * self.pr.samp_sr)]

        audio = self.wave2spec(audio)
        x = self.net(audio)
        if return_angle: 
            pred = self.pred_head(x).squeeze(-1) 
            return x, pred
        return x
    
    def construct_audio_net(self, args, pr):
        in_channels = 4
        if args.audio_backbone == 'resnet10':
            model = torchvision.models.resnet._resnet(torchvision.models.resnet.BasicBlock, [1, 1, 1, 1], weights=None, progress=False)
        elif args.audio_backbone == 'resnet18':
            model = torchvision.models.resnet18(weights=None)
        elif args.audio_backbone == 'resnet34':
            model = torchvision.models.resnet34(weights=None)
        elif args.audio_backbone == 'resnet50':
            model = torchvision.models.resnet50(weights=None)

        model.conv1 = torch.nn.Conv2d(in_channels, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
        model.fc = nn.Linear(model.fc.in_features, pr.audio_feature_dim)
        
        # Initialize weights
        for m in model.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, nn.Linear):
                nn.init.trunc_normal_(m.weight, mean=0.0, std=0.01)
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.normal_(m.weight, mean=1, std=0.02)
                nn.init.constant_(m.bias, 0)
        return model
    
    def wave2spec(self, wave, return_complex=False):
        '''
            return normalized magnitude and phase spectrogram: (N, C, F, T), C = 4 with both magnitude and phase
        '''
        # import pdb; pdb.set_trace()
        N, C, L = wave.shape
        wave = wave.view(N * C, -1)

        frames = 256
        hop_length = int(L // (frames - 1))

        spec = torch.stft(
            input=wave,
            n_fft=self.pr.n_fft,
            hop_length=hop_length,
            win_length=self.pr.win_length,
            return_complex=True
        )

        spec = spec.contiguous().view(N, C, *spec.shape[1:])
        if return_complex:
            return spec

        if self.args.use_real_imag:
            spec = torch.view_as_real(spec)
        else:
            mag, phase = spec.abs().unsqueeze(-1), spec.angle().unsqueeze(-1)
            mag = self.normalize_magnitude(mag)
            phase = self.normalize_phase(phase)
            spec = torch.cat([mag, phase], dim=-1)
        spec = spec.permute(0, 1, 4, 2, 3)
        spec = spec.contiguous().view(N, -1, *spec.shape[3:])
        spec = spec[:, :, :-1, :frames]
        return spec

    def normalize_magnitude(self, spec):
        # import pdb; pdb.set_trace()
        spec_min = -100
        spec_max = 60
        spec = torch.maximum(spec, torch.tensor(self.pr.log_offset))
        spec = 20 * torch.log10(spec)
        spec = (spec - spec_min) / (spec_max - spec_min) * 2 - 1
        spec = torch.clip(spec, -1.0, 1.0)
        return spec

    def normalize_phase(self, phase):
        pi = 3.1416
        phase = phase / pi
        phase = torch.clip(phase, -1.0, 1.0)
        return phase


class AudioLoCNet(nn.Module):
    # Sound Localization Net
    def __init__(self, args, pr, net):
        super(AudioLoCNet, self).__init__()
        self.pr = pr
        self.args = args
        self.net = net
    
        self.azimuth_loss_type = args.azimuth_loss_type
        if self.azimuth_loss_type == 'classification':
            self.criterion = nn.CrossEntropyLoss(reduction='none')
            num_classes = pr.num_classes
        elif self.azimuth_loss_type == 'regression':
            self.criterion = nn.L1Loss(reduction='none')
            num_classes = 1

        self.fc = nn.Linear(pr.audio_feature_dim, num_classes)
        nn.init.trunc_normal_(self.fc.weight, mean=0.0, std=0.01)

        if args.freeze_audio:
            for param in self.net.parameters():
                param.requires_grad = False
            tqdm.write('Freezed the audio backbone.')

    def forward(self, inputs, loss=False, evaluate=False):
        # import pdb; pdb.set_trace()
        augment = not evaluate
        audio_1 = inputs['audio_1']
        audio_feat = self.net(audio_1, augment=augment)
        preds = self.fc(audio_feat)

        if loss:
            loss = self.calc_loss(preds, inputs)
            return loss
        if evaluate:
            output = self.evaluate(preds, inputs)
            return output

        return preds

    def calc_loss(self, preds, inputs):
        if self.azimuth_loss_type == 'classification':
            target = inputs['angle_bin_between_source1_camera1']
            loss = self.criterion(preds, target.long())
        elif self.azimuth_loss_type == 'regression':
            # normalize the prediction to angle range
            preds = torch.tanh(preds.squeeze(-1)) * 180.0
            target = inputs['angle_between_source1_camera1']
            loss = self.criterion(preds, target.long())
        return loss

    def evaluate(self, preds, inputs):
        loss = self.calc_loss(preds, inputs)
        output = {
            'Loss': loss.view(1, -1)
        }
        if self.azimuth_loss_type == 'classification':
            target = inputs['angle_bin_between_source1_camera1']
            top_1_acc = torch_utils.calc_acc(preds, target, k=1)
            output['Top-1 acc'] = top_1_acc.view(1, -1)
        elif self.azimuth_loss_type == 'regression':
            # place holder for main.py
            pass
        return output

    def score_model_performance(self, res):
        if self.azimuth_loss_type == 'classification':
            score = res['Top-1 acc']
        elif self.azimuth_loss_type == 'regression':
            score = 1 / res['Loss']
        return score
