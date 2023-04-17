import torch
import math
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models
import torchaudio
import numpy as np

import sys
sys.path.append('..')
from config import init_args, params
import models
from models import FloatEmbeddingSine



class SLfMNet(nn.Module):
    # SLfM pretext net
    def __init__(self, args, pr, nets):
        super(SLfMNet, self).__init__()
        self.pr = pr
        self.args = args
        self.n_view = args.n_view
        self.no_vision = args.no_vision
        self.no_cond_audio = args.no_cond_audio
        self.add_geometric = args.add_geometric
        self.use_gt_rotation = args.use_gt_rotation
        self.generative_loss_ratio = args.generative_loss_ratio

        self.vision_net, self.audio_net = nets
        self.generative_net = models.__dict__[pr.generative_net](args, pr)

        if self.use_gt_rotation:
            self.rota_embedding = FloatEmbeddingSine(num_pos_feats=pr.visual_feature_dim, scale=1)

        self.freeze_param(args)


    def forward(self, inputs, loss=False, evaluate=False, inference=False):
        # import pdb; pdb.set_trace()
        augment = not (evaluate or inference)
        cond_audios, audio_input, audio_output = self.generate_audio_pair(inputs)
        cond_feats = self.encode_conditional_feature(inputs, cond_audios, augment)
        # import pdb; pdb.set_trace()
        pred_audio = self.generative_net(audio_input, cond_feats)

        if loss:
            loss = self.calc_loss(inputs, audio_output, pred_audio)
            return loss
        if evaluate:
            output = self.calc_loss(inputs, audio_output, pred_audio, evaluate=True)
            return output
        if inference:
            output = self.inference(inputs, pred_audio)
            return output
        
        return pred_audio
    

    def calc_loss(self, inputs, target_audio, pred_audio, evaluate=False):
        output = {}
        N = pred_audio.shape[0]
        spec_weight = 1
        if self.args.loss_type == 'L1':
            spec_loss = F.l1_loss(pred_audio, target_audio, reduction='none')
            spec_weight = 10
        elif self.args.loss_type == 'L2':
            spec_loss = F.mse_loss(pred_audio, target_audio, reduction='none')

        spec_loss = spec_loss.view(N, -1).mean(dim=-1)
        spec_loss = spec_loss.view(-1, self.n_view - 1).mean(dim=-1)
        spec_loss = spec_weight * spec_loss 
        output['Spec Loss'] = spec_loss
        loss = spec_loss * self.generative_loss_ratio
        output['Loss'] = loss
        if evaluate:
            return output
        return loss


    def inference(self, inputs, pred_audio):
        # import pdb; pdb.set_trace()
        gt_audio = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(1, self.n_view)]
        gt_audio = torch.cat(gt_audio, dim=1)
        audio_shape = gt_audio.shape 
        gt_audio = gt_audio.contiguous().view(-1, *audio_shape[2:])
        c = int(gt_audio.shape[1] // 2)

        if self.args.mono2binaural:
            audio_mix = gt_audio[:, :c, :] + gt_audio[:, c:, :]
            audio_input = self.wave2spec(audio_mix, return_complex=True).squeeze().detach()
            pred_audio = pred_audio.permute(0, 2, 3, 1)
            pred_audio = torch.view_as_complex(pred_audio.contiguous())
            pred_audio = torch.cat([pred_audio, audio_input[:, -1:, ...]], dim=1)
        
            pred_audio = torch.istft(
                input=pred_audio,
                n_fft=self.pr.n_fft,
                hop_length=self.pr.hop_length,
                win_length=self.pr.win_length
            ).unsqueeze(1)
            pred_left = (audio_mix + pred_audio) / 2
            pred_right = (audio_mix - pred_audio) / 2
            pred_audio = torch.cat([pred_left, pred_right], dim=1)
        else:
            raise NotImplementedError
        
        return {
            'pred_wave': pred_audio,
            'gt_wave': gt_audio
        }
    

    def encode_conditional_feature(self, inputs, cond_audio, augment):
        # import pdb; pdb.set_trace()
        # We always set the Img 1 as conditional view
        B = cond_audio.shape[0]

        # ------  Encode the conditional audio at the source viewpoint  --------- #
        if self.no_cond_audio:
            cond_audio_feat = None
        else:
            cond_audio_feat = self.audio_net(cond_audio, augment=augment)
            cond_audio_feat = torch.cat([cond_audio_feat.unsqueeze(1)] * (self.n_view - 1), dim=1)
            cond_audio_feat = cond_audio_feat.contiguous().view(-1, *cond_audio_feat.shape[2:])

        # ------  Encode the relative camera pose between different view  --------- #
        if self.no_vision:
            im_features = None
        else:
            single_im_features = []
            for i in range(0, self.n_view):
                im_feature = self.vision_net.forward_backbone(inputs[f'img_{i+1}'], augment=augment)
                single_im_features.append(im_feature)

            im_features = []
            for i in range(1, self.n_view):
                corr_feature = self.vision_net.forward_correlation(single_im_features[i], single_im_features[0])
                im_features.append(corr_feature.unsqueeze(1))

            im_features = torch.cat(im_features, dim=1)
            im_features = im_features.contiguous().view(-1, *im_features.shape[2:])

            if self.use_gt_rotation:
                theta = torch.cat([inputs[f'relative_camera{i}_angle'].unsqueeze(1) for i in range(1, self.n_view)], dim=1)
                theta = theta.contiguous().view(theta.shape[0] * theta.shape[1], -1)
                theta = theta / 180.0 * math.pi
                im_features = self.rota_embedding(theta.float()).detach()

        # ------  Concat the conditional features  --------- #
        if self.no_vision and not self.no_cond_audio:
            cond_feats = cond_audio_feat
        elif not self.no_vision and self.no_cond_audio:
            cond_feats = im_features
        elif not self.no_vision and not self.no_cond_audio:
            cond_feats = torch.cat([cond_audio_feat, im_features], dim=-1)
        else:
            cond_feats = None

        return cond_feats
    
    def generate_audio_pair(self, inputs):
        # import pdb; pdb.set_trace()
        target_view_audio = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(1, self.n_view)]
        target_view_audio = torch.cat(target_view_audio, dim=1)
        audio_shape = target_view_audio.shape 
        target_view_audio = target_view_audio.contiguous().view(-1, *audio_shape[2:])

        c = int(target_view_audio.shape[1] // 2)
        if self.args.mono2binaural:
            audio_mix = (target_view_audio[:, :c, :] + target_view_audio[:, c:, :])
            audio_input = self.wave2spec(audio_mix, return_real_imag=True).detach()
            audio_diff = (target_view_audio[:, :c, :] - target_view_audio[:, c:, :])
            audio_output = self.wave2spec(audio_diff, return_real_imag=True).detach()
        else:
            audio_input = self.wave2spec(target_view_audio[:, :c, :], return_mag_phase=True).detach()
            audio_output = self.wave2spec(target_view_audio[:, c:, :], return_mag_phase=True).detach()
        cond_audio = inputs['audio_1']
        return cond_audio, audio_input, audio_output


    def wave2spec(self, wave, return_complex=False, return_real_imag=False, return_mag_phase=False):
        # import pdb; pdb.set_trace()
        N, C, _ = wave.shape
        wave = wave.view(N * C, -1)
        spec = torch.stft(
            input=wave,
            n_fft=self.pr.n_fft,
            hop_length=self.pr.hop_length,
            win_length=self.pr.win_length,
            return_complex=True
        )
        spec = spec.contiguous().view(N, C, *spec.shape[1:])
        if return_complex:
            return spec
        elif return_real_imag:
            spec = torch.view_as_real(spec)
            spec = spec.permute(0, 1, 4, 2, 3)
            spec = spec.view(N, -1, *spec.shape[3:])
        elif return_mag_phase:
            mag, phase = spec.abs().unsqueeze(-1), spec.angle().unsqueeze(-1)
            mag = self.normalize_magnitude(mag)
            phase = self.normalize_phase(phase)
            spec = torch.cat([mag, phase], dim=-1)
            spec = spec.permute(0, 1, 4, 2, 3)
            spec = spec.contiguous().view(N, -1, *spec.shape[3:])
        else:
            # return log magnitude
            spec = spec.abs()
            spec = self.normalize_magnitude(spec)
        # spec: (N, C, F-1, T)
        spec = spec[:, :, :-1, :]
        return spec


    def normalize_magnitude(self, spec, inverse=False):
        # import pdb; pdb.set_trace()
        spec_min = -100
        spec_max = 60
        if not inverse:
            spec = torch.maximum(spec, torch.tensor(self.pr.log_offset))
            spec = 20 * torch.log10(spec)
            spec = (spec - spec_min) / (spec_max - spec_min) * 2 - 1
            spec = torch.clip(spec, -1.0, 1.0)
            # spec = torch.log(spec + self.pr.log_offset)
        else:
            spec = (spec + 1) / 2
            spec = spec * (spec_max - spec_min) + spec_min
            spec = 10 ** (spec / 20)
        return spec


    def normalize_phase(self, phase, inverse=False):
        pi = 3.1416
        if not inverse:
            phase = phase / pi
            phase = torch.clip(phase, -1.0, 1.0)
        else:
            phase = phase * pi
        return phase


    def freeze_param(self, args):
        if args.freeze_camera:
            for param in self.vision_net.parameters():
                param.requires_grad = False
        if args.freeze_audio:
            for param in self.audio_net.parameters():
                param.requires_grad = False
        if args.freeze_generative:
            for param in self.generative_net.parameters():
                param.requires_grad = False


    def score_model_performance(self, res):
        score = 1 / res['Loss']
        return score
