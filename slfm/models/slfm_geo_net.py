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
from models import SLfMNet



class SLfMGeoNet(SLfMNet):
    # Multi-view Audio SFM Net
    def __init__(self, args, pr, nets):
        super(SLfMGeoNet, self).__init__(args, pr, nets)
        assert self.add_geometric == True, 'Geometric feature needs to be enabled.'
        self.activation = args.activation
        self.use_gt_rotation = args.use_gt_rotation
        self.geometric_loss_ratio = args.geometric_loss_ratio * 2 # there is a factor between code implementation and paper formualtion
        self.binaural_loss_ratio = args.binaural_loss_ratio
        self.symmetric_loss_ratio = args.symmetric_loss_ratio


    def forward(self, inputs, loss=False, evaluate=False, inference=False):
        # import pdb; pdb.set_trace()
        augment = not (evaluate or inference)
        cond_audios, audio_input, audio_output = self.generate_audio_pair(inputs)
        features = self.encode_conditional_feature(inputs, cond_audios, augment)
        # import pdb; pdb.set_trace()
        pred_audio = self.generative_net(audio_input, features['cond_feats'])

        if loss:
            loss = self.calc_loss(inputs, features, audio_output, pred_audio, augment)
            return loss
        if evaluate:
            output = self.calc_loss(inputs, features, audio_output, pred_audio, evaluate=True)
            return output
        if inference:
            output = self.inference(inputs, pred_audio)
            return output
        
        return pred_audio
    

    def calc_loss(self, inputs, features, target_audio, pred_audio, augment=False, evaluate=False):
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

        if self.geometric_loss_ratio > 0:
            if not self.args.filter_sound and self.args.sound_permutation:
                geometric_loss = self.calc_geometric_loss_with_permutation(inputs, features, augment=augment)
            else:
                geometric_loss = self.calc_geometric_loss(inputs, features, augment=augment)
        else:
            geometric_loss = torch.zeros_like(spec_loss)

        if self.binaural_loss_ratio > 0:
            binaural_loss = self.calc_binaural_loss(inputs, features)
        else:
            binaural_loss = torch.zeros_like(spec_loss)
        
        if self.symmetric_loss_ratio > 0:
            audio_sym_loss = self.calc_audio_symmetric_loss(inputs, features)
            rota_sym_loss = self.calc_rota_symmetric_loss(inputs, features)
        else:
            audio_sym_loss = torch.zeros_like(spec_loss)
            rota_sym_loss = torch.zeros_like(spec_loss)
        
        output['Geometric Loss'] = geometric_loss
        output['Binaural Loss'] = binaural_loss
        output['Audio Symmetric Loss'] = audio_sym_loss
        output['Rotation Symmetric Loss'] = rota_sym_loss
        
        loss = self.generative_loss_ratio * spec_loss + self.geometric_loss_ratio * geometric_loss + self.binaural_loss_ratio * binaural_loss + self.symmetric_loss_ratio * (audio_sym_loss + rota_sym_loss)
        output['Loss'] = loss
        
        if evaluate:
            return output
        return loss


    def encode_conditional_feature(self, inputs, ref_audio, augment):
        # import pdb; pdb.set_trace()
        # We always set the Img 1 as the source view
        single_im_features = []
        for i in range(0, self.n_view):
            im_feature = self.vision_net.forward_backbone(inputs[f'img_{i+1}'], augment=augment)
            single_im_features.append(im_feature)

        rots = []
        rots_inv = []
        im_features = []
        im_features_inv = []

        for i in range(1, self.n_view):
            # the order between features should be i -> 0 for our model
            corr_feature, rot = self.vision_net.forward_correlation(single_im_features[i], single_im_features[0], return_angle=True)
            corr_feature_inv, rot_inv = self.vision_net.forward_correlation(single_im_features[0], single_im_features[i], return_angle=True)
            im_features.append(corr_feature.unsqueeze(1))
            im_features_inv.append(corr_feature_inv.unsqueeze(1))
            rots.append(rot.unsqueeze(1))
            rots_inv.append(rot_inv.unsqueeze(1))

        im_features = torch.cat(im_features, dim=1)
        im_features = im_features.contiguous().view(-1, *im_features.shape[2:])
        im_features_inv = torch.cat(im_features_inv, dim=1)
        im_features_inv = im_features_inv.contiguous().view(-1, *im_features_inv.shape[2:])
        rots = torch.cat(rots, dim=1)
        rots = rots.contiguous().view(rots.shape[0] * rots.shape[1], -1).squeeze(-1)
        rots_inv = torch.cat(rots_inv, dim=1)
        rots_inv = rots_inv.contiguous().view(rots_inv.shape[0] * rots_inv.shape[1], -1).squeeze(-1)
        
        ref_audio_feat, ref_angle = self.audio_net(ref_audio, return_angle=True, augment=augment)
        ref_audio_feat = torch.cat([ref_audio_feat.unsqueeze(1)] * (self.n_view - 1), dim=1)
        ref_audio_feat = ref_audio_feat.contiguous().view(-1, *ref_audio_feat.shape[2:])
        ref_angle = torch.cat([ref_angle.unsqueeze(1)] * (self.n_view - 1), dim=1)
        ref_angle = ref_angle.contiguous().view(ref_angle.shape[0] * (self.n_view - 1), -1).squeeze(-1)

        target_angles = []
        for i in range(1, self.n_view):
            _, target_angle = self.audio_net(inputs[f'audio_{i+1}'], return_angle=True, augment=augment)
            target_angles.append(target_angle.unsqueeze(1))
        target_angles = torch.cat(target_angles, dim=1)
        target_angles = target_angles.contiguous().view(target_angles.shape[0] * target_angles.shape[1], -1).squeeze(-1)

        cond_feats = torch.cat([ref_audio_feat, im_features], dim=-1)
        return {
            'cond_feats': cond_feats, 
            'ref_angle': ref_angle,
            'target_angles': target_angles,
            'rot': rots,
            'rot_inv': rots_inv
        }

    def calc_geometric_loss(self, inputs, features, augment):
        '''
            We project 3D space into 2D space and solve the geometric problem (x, y)
            The geometric loss ensure the geometirc principle between them while it doesn't regularize the 
            sound source direction. 
            here camera rotation we difine right as -, left as +
            sound direction, left as +, right as -
        '''
        # import pdb; pdb.set_trace()
        if self.use_gt_rotation:
            theta = torch.cat([inputs[f'relative_camera{i}_angle'].unsqueeze(1) for i in range(1, self.n_view)], dim=1)
            theta = theta.contiguous().view(theta.shape[0] * theta.shape[1], -1).squeeze(-1)
            theta = theta / 180.0 * math.pi
            theta = - theta.float().detach()
        else:
            theta = self.rot2theta(features['rot'])
            theta = - theta                       # the sound rotation is symmetric to the agent rotation 

        rots = [torch.cos(theta).unsqueeze(-1), -torch.sin(theta).unsqueeze(-1), 
                torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)]
        rots = torch.cat(rots, dim=-1)
        rots = rots.contiguous().view(-1, 2, 2)

        ref_angle = self.logit2angle(features['ref_angle'])
        ref_vec = [torch.cos(ref_angle).unsqueeze(-1), torch.sin(ref_angle).unsqueeze(-1)]
        ref_vec = torch.cat(ref_vec, dim=-1)

        target_angles = self.logit2angle(features['target_angles'])
        target_vec = [torch.cos(target_angles).unsqueeze(-1), torch.sin(target_angles).unsqueeze(-1)]
        target_vec = torch.cat(target_vec, dim=-1)

        rotated_ref_vec = torch.matmul(rots, ref_vec.unsqueeze(-1)).squeeze(-1)
        dot_product = (rotated_ref_vec * target_vec).sum(-1)
        dot_product_target = torch.ones_like(dot_product)
        geometric_loss = F.l1_loss(dot_product, dot_product_target, reduction='none')
        geometric_loss = geometric_loss.view(-1, self.n_view - 1).mean(dim=-1)
        return geometric_loss
    
    def calc_geometric_loss_with_permutation(self, inputs, features, augment):
        '''
            This loss is for full angle setup only. Since full angle setup has sound ambiguity, we add permutation combination to see if the model could overcome the ambuguity.
        '''
        # import pdb; pdb.set_trace()
        if self.use_gt_rotation:
            theta = torch.cat([inputs[f'relative_camera{i}_angle'].unsqueeze(1) for i in range(1, self.n_view)], dim=1)
            theta = theta.contiguous().view(theta.shape[0] * theta.shape[1], -1).squeeze(-1)
            theta = theta / 180.0 * math.pi
            theta = - theta.float().detach()
        else:
            theta = self.rot2theta(features['rot'])
            theta = - theta                       # the sound rotation is symmetric to the agent rotation 

        if self.args.inverse_camera: # we study the inversed camera prediction for ambiguity 
            theta = - theta

        rots = [torch.cos(theta).unsqueeze(-1), -torch.sin(theta).unsqueeze(-1), 
                torch.sin(theta).unsqueeze(-1), torch.cos(theta).unsqueeze(-1)]
        rots = torch.cat(rots, dim=-1)
        rots = rots.contiguous().view(-1, 2, 2)

        ref_angle = self.logit2angle(features['ref_angle'])
        ref_angle_inv = self.inverse_sound_direction(ref_angle)
        ref_angle = torch.cat([
            ref_angle.unsqueeze(-1), ref_angle_inv.unsqueeze(-1),
            ref_angle.unsqueeze(-1), ref_angle_inv.unsqueeze(-1)], dim=-1)
        ref_angle = ref_angle.view(-1)
        
        target_angles = self.logit2angle(features['target_angles'])
        target_angles_inv = self.inverse_sound_direction(target_angles)
        target_angles = torch.cat([
            target_angles.unsqueeze(-1), target_angles_inv.unsqueeze(-1),
            target_angles_inv.unsqueeze(-1), target_angles.unsqueeze(-1)], dim=-1)
        target_angles = target_angles.view(-1)
        
        ref_vec = [torch.cos(ref_angle).unsqueeze(-1), torch.sin(ref_angle).unsqueeze(-1)]
        ref_vec = torch.cat(ref_vec, dim=-1)

        target_vec = [torch.cos(target_angles).unsqueeze(-1), torch.sin(target_angles).unsqueeze(-1)]
        target_vec = torch.cat(target_vec, dim=-1)

        rots = rots.repeat_interleave(4, dim=0) # match the permutation number

        rotated_ref_vec = torch.matmul(rots, ref_vec.unsqueeze(-1)).squeeze(-1)
        dot_product = (rotated_ref_vec * target_vec).sum(-1)
        dot_product_target = torch.ones_like(dot_product)
        geometric_loss = F.l1_loss(dot_product, dot_product_target, reduction='none')
        geometric_loss = geometric_loss.view(geometric_loss.shape[0] // 4, 4, -1).mean(dim=-1)
        geometric_loss = geometric_loss.min(dim=-1)[0]
        geometric_loss = geometric_loss.view(-1, self.n_view - 1).mean(dim=-1)
        return geometric_loss

    def calc_binaural_loss(self, inputs, features):
        '''
            We calcualte the binaural cue loss with a weak supervision: whether sound is on the left or right
            sound direction: left as +, right as -
        '''
        # import pdb; pdb.set_trace()
        ref_angle = self.logit2angle(features['ref_angle'])
        ref_angles = ref_angle[::(self.n_view - 1)]
        target_angles = self.logit2angle(features['target_angles'])
        target_angles = target_angles.view(-1, self.n_view - 1)
        angles = torch.cat([ref_angles.unsqueeze(-1), target_angles], dim=-1)
        angles = (torch.sin(angles) + 1) / 2
        audios = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(self.n_view)]
        audios = torch.cat(audios, dim=1)
        # Re-cut the conditional audio clip to meet the requirement
        audios = audios[..., :int(self.pr.cond_clip_length * self.pr.samp_sr)]

        # advanced IID cues 
        audios = audios.contiguous().view(-1, *audios.shape[2:])
        audios = self.audio_net.wave2spec(audios, return_complex=True).abs()
        ild_cues = torch.log(audios[:, 0, :, :].mean(dim=-2) / audios[:, 1, :, :].mean(dim=-2))
        ild_cues = torch.sign(ild_cues).sum(dim=-1)
        ild_cues = torch.sign(ild_cues)
        ild_cues = ild_cues.view(-1, self.n_view)
        target = (ild_cues + 1) / 2

        loss = F.binary_cross_entropy(angles, target.detach(), reduction='none').mean(-1)
        return loss

    def calc_audio_symmetric_loss(self, inputs, features):
        # import pdb; pdb.set_trace()
        ref_angle = self.logit2angle(features['ref_angle'])
        ref_angles = ref_angle[::(self.n_view - 1)]
        target_angles = self.logit2angle(features['target_angles'])
        target_angles = target_angles.view(-1, self.n_view - 1)
        angles = torch.cat([ref_angles.unsqueeze(-1), target_angles], dim=-1)

        flipped_audios = [inputs[f'audio_{i+1}'].unsqueeze(1) for i in range(self.n_view)]
        flipped_audios = torch.cat(flipped_audios, dim=1)
        flipped_audios = flipped_audios.contiguous().view(-1, *flipped_audios.shape[2:]).flip(1)
        _, flipped_angles = self.audio_net(flipped_audios, return_angle=True)
        flipped_angles = flipped_angles.view(-1, self.n_view)
        flipped_angles = self.logit2angle(flipped_angles)
        angle_sum = angles + flipped_angles
        target = torch.zeros_like(angle_sum)
        loss = F.l1_loss(angle_sum, target.detach(), reduction='none').mean(-1)
        return loss

    def calc_rota_symmetric_loss(self, inputs, features):
        theta = self.rot2theta(features['rot'])
        theta_inv = self.rot2theta(features['rot_inv'])
        rota_sum = theta + theta_inv
        target = torch.zeros_like(rota_sum)
        loss = F.l1_loss(rota_sum, target.detach(), reduction='none').mean(-1)
        return loss

    def logit2angle(self, logit, inverse=False):
        '''
            map audio logit prediction to angle 
        '''
        angle_range = 2 if self.args.filter_sound else 1
        if self.activation == 'tanh':
            theta_s = torch.tanh(logit)
        elif self.activation == 'clamp':
            theta_s = torch.clamp(logit, min=-1., max=1.)
        elif self.activation == 'sigmoid':
            theta_s = torch.sigmoid(logit)
            theta_s = theta_s * 2. - 1.
        theta_s = theta_s * math.pi / angle_range
        return theta_s
    
    def rot2theta(self, rot, inverse=False):
        '''
            map rotation logit prediction to angle
        '''
        # activation each prediction
        if self.activation == 'tanh':
            theta = torch.tanh(rot)
        elif self.activation == 'clamp':
            theta = torch.clamp(rot, min=-1., max=1.)
        elif self.activation == 'sigmoid':
            theta = torch.sigmoid(rot)
            theta = theta * 2. - 1.
        # fit the angle prediction into corresponding range
        if self.args.finer_rotation:
            theta = theta * math.pi / 2       # Finer range: [-90, 90]
        else:
            theta = theta * math.pi           # Full range: [-180, 180]
        return theta


    def inverse_sound_direction(self, pred):
        pred_inv = torch.clone(pred)
        pred_inv[pred_inv <= 0.] = - math.pi - pred_inv[pred_inv <= 0]
        pred_inv[pred_inv > 0.] = math.pi - pred_inv[pred_inv > 0]
        return pred_inv


    def freeze_param(self, args):
        # import pdb; pdb.set_trace()
        if args.freeze_camera:
            for param in self.vision_net.parameters():
                param.requires_grad = False
        if args.freeze_audio:
            for param in self.audio_net.parameters():
                param.requires_grad = False
        if args.freeze_generative:
            for param in self.generative_net.parameters():
                param.requires_grad = False
        
        # Unfreeze the parameter we need to optimize
        for param in self.vision_net.rot_head.parameters():
            param.requires_grad = True
        for param in self.audio_net.pred_head.parameters():
            param.requires_grad = True


    def train(self, mode=True):
        if not isinstance(mode, bool):
            raise ValueError("training mode is expected to be boolean")
        self.training = mode
        for module in self.children():
            module.train(mode)
        
        if mode:
            if self.args.freeze_camera:
                for m in self.vision_net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
                        
            if self.args.freeze_audio:
                for m in self.audio_net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()

            if self.args.freeze_generative:
                for m in self.generative_net.modules():
                    if isinstance(m, nn.BatchNorm2d):
                        m.eval()
        return self

    def score_model_performance(self, res):
        # IMPORTANT: you need to design geometric_weight here
        geometric_weight = 100.0
        score = 1 / (
            self.generative_loss_ratio * res['Spec Loss'] +
            geometric_weight * res['Geometric Loss'] +
            self.binaural_loss_ratio * res['Binaural Loss'] + 
            self.symmetric_loss_ratio * (res['Audio Symmetric Loss'] + res['Rotation Symmetric Loss'])
        )
        return score








