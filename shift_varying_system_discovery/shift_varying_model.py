import time 

import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as init

import matplotlib.pyplot as plt

from measurements import ShiftVaryingBlur

class ShiftVaryingSystemCNNV1(nn.Module):
    def __init__(self, img_height = 256, img_width = 256, num_psfs = 2, kernel_size = 33):
        super(ShiftVaryingSystemCNNV1, self).__init__()
        self.num_psfs = num_psfs

        filters = torch.zeros(num_psfs, 1, kernel_size, kernel_size).float()
        filters[:, 0, kernel_size // 2, kernel_size // 2] = 1.0  # Set the center to 1
        self.filters = nn.Parameter(filters)  # Convert to nn.Parameter

        self.weights = nn.Parameter(torch.ones(num_psfs, img_height, img_width).float() / num_psfs)

    def forward(self, x):
        batch_size, c, h, w = x.shape

        weighted_input = x.unsqueeze(2) * self.weights.unsqueeze(0).unsqueeze(1)  # Broadcast the weights for all batches

        # Initialize list to store the results for each PSF
        psf_outputs = []

        for i in range(self.num_psfs):
            filter =  self.filters[i].repeat(c, 1, 1, 1)  # Shape becomes [3, 1, 33, 33]

            psf_output = F.conv2d(weighted_input[:,:,i], filter, padding='same', groups=c)
            psf_outputs.append(psf_output)

        psf_outputs = torch.stack(psf_outputs, dim=1).sum(axis=1)  # Shape: (batch_size, num_psfs, c, h, w)

        return psf_outputs

class WeightEnforcer(nn.Module):
    def __init__(self, weight_low_rank_factors, weight_scale_factor, img_size = 256, banded_matrix_width = -1):
        super().__init__()
        self.img_size = img_size
        self.weight_scale_factor = weight_scale_factor
        self.weight_low_rank_factors = weight_low_rank_factors
        self.banded_matrix_width = banded_matrix_width
        self.use_banded_matrix_width = banded_matrix_width > 0

    def forward(self, pre_weights, x = None):
        if pre_weights is None:
            return 1.
        
        if not self.use_banded_matrix_width:
            if self.weight_low_rank_factors <= 0:
                weights = pre_weights
            else:
                weights = torch.matmul(pre_weights[0], pre_weights[1]) / self.weight_low_rank_factors
            
            weights = weights.unsqueeze(0).unsqueeze(1)  # Shape: 1x1x(num_psfs-1)xHxW
            last_weight = 1 - weights.sum(dim=2, keepdim=True)  # Enforce sum to 1
            all_weight = torch.cat([weights, last_weight], dim=2) # Shape: 1x1xnum_psfsxHxW
            
            if self.weight_scale_factor > 1:
                # pad_size = 4  # Adjust based on your needs
                # padded_weight = F.pad(all_weight, (pad_size, pad_size, pad_size, pad_size), mode="replicate")
                # all_weight = all_weight[..., pad_size:-pad_size, pad_size:-pad_size]
                all_weight = F.interpolate(
                    all_weight.squeeze(0), size=self.img_size, 
                    mode="bilinear",  align_corners=False
                ).unsqueeze(0)
           
            if x is None:
                return all_weight

            b,c,h,w = x.shape
            num_psfs = all_weight.shape[2]
            
            weighted_input = x.unsqueeze(2) * all_weight  # (batch_size, c, num_psfs, h, w)
            
            return  weighted_input.reshape(b * c , num_psfs, h, w)
        
        else:
            if x is None: 
                return pre_weights
            
            b_w = 2 * self.banded_matrix_width + 1
            b,c,h,w = x.shape

            # extract unfold 
            x_padded = F.pad(x, (b_w//2, b_w//2))
            x_unfold = F.unfold(x_padded, kernel_size=(1,b_w),stride=1)
            
            x_unfold = x_unfold.view(b, c, b_w, h * w).permute(0, 1, 3, 2)  # (b, c, h*w, b_w)
            x_unfold = x_unfold.reshape(b * c, h * w, b_w)  # (b*c, h*w, b_w)

            if self.weight_scale_factor > 1:
                d = h // self.weight_scale_factor
                weight = pre_weights.reshape(pre_weights.shape[0],d,d,b_w).permute(0, 3, 1, 2)
                
                weight = F.interpolate(weight, size=(h, w), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                pre_weights = weight.reshape(pre_weights.shape[0], -1 , b_w)
                # print("weigth shape ", x_unfold.shape, pre_weights.shape)
            # print("pre_weights ", pre_weights.shape, d,h)
            # print("pre_weights ", .shape)

            output = torch.einsum('bpd,npd->bnp', x_unfold, pre_weights)

            output = output.view(b * c, pre_weights.shape[0], h, w)

            return output            

import torch
import torch.nn as nn

class MultiLayerShiftVaryingSystem(nn.Module):
    def __init__(self, num_layers=3, img_height=256, img_width=256, num_psfs=2, kernel_size=33, device='cuda', 
                 learn_w=False, weight_low_rank_factors=-1, non_blind=False, conv_fft=False, weight_scale_factor=1, 
                 weight_init_func='constant', kernel_init_func='delta', banded_matrix_width=-1):
        super(MultiLayerShiftVaryingSystem, self).__init__()

        self.num_layers = num_layers

        # If num_psfs is a single int, replicate it for all layers
        if isinstance(num_psfs, int):
            num_psfs = [num_psfs] * num_layers
        elif isinstance(num_psfs, list):
            assert len(num_psfs) == num_layers, "Length of num_psfs list must equal num_layers"
        else:
            raise TypeError("num_psfs must be an int or a list of ints")

        self.layers = nn.ModuleList([
            ShiftVaryingSystemCNN(
                img_height=img_height,
                img_width=img_width,
                num_psfs=num_psfs[i],
                kernel_size=kernel_size,
                device=device,
                learn_w=learn_w,
                weight_low_rank_factors=weight_low_rank_factors,
                non_blind=non_blind,
                conv_fft=conv_fft,
                weight_scale_factor=weight_scale_factor,
                weight_init_func=weight_init_func,
                kernel_init_func=kernel_init_func,
                banded_matrix_width=banded_matrix_width
            ) for i in range(num_layers)
        ])

    def forward(self, x):
        out = x
        for layer in self.layers:
            out = layer(out) + out  # skip connection
        return out
        
    def adjoint(self, x):
        out = x
        for layer in reversed(self.layers):
            out = layer.adjoint(out) + out  # adjoint of skip connection
        return out

    def weight_norm(self, p=1):
        return sum(layer.weight_norm(p=p) for layer in self.layers)

    def filter_norm(self, p=1):
        return sum(layer.filter_norm(p=p) for layer in self.layers)


    def get_filters(self):
        return [layer.filters for layer in self.layers]

    def get_weights(self):
        return [layer.weights for layer in self.layers]

class ShiftVaryingSystemCNN(nn.Module):
    def __init__(self, img_height = 256, img_width = 256, num_psfs = 2, kernel_size = 33, device = 'cuda', learn_w = False, 
                 weight_low_rank_factors = -1, non_blind = False, conv_fft = False, weight_scale_factor = 1, 
                 weight_init_func = 'constant', kernel_init_func = 'delta', banded_matrix_width = -1):
        super(ShiftVaryingSystemCNN, self).__init__()
        self.img_height = img_height
        self.img_width = img_width
        self.num_psfs = num_psfs
        self.kernel_size = kernel_size
        self.device = device
        self.learn_w = learn_w
        self.weight_low_rank_factors = weight_low_rank_factors
        self.eval_mode = False
        self.non_blind = non_blind
        self.conv_fft = conv_fft or kernel_size > 330
        self.true_filters = None
        self.weight_scale_factor = weight_scale_factor
        self.weight_init_func = weight_init_func
        self.kernel_init_func = kernel_init_func

        self.banded_matrix_width = banded_matrix_width
        self.use_banded_matrix = self.banded_matrix_width > 0
        
        self.weight_enforcer = WeightEnforcer(weight_low_rank_factors = self.weight_low_rank_factors, 
                                              weight_scale_factor = self.weight_scale_factor,
                                              img_size = self.img_height, banded_matrix_width = self.banded_matrix_width)
        
        # Define parameters
        self.filters = nn.Parameter(
            torch.zeros(num_psfs, 1, kernel_size, kernel_size, device=self.device, requires_grad=True)
        ) if not self.non_blind else None
        
        if self.num_psfs < 2:    
            # self.weights.requires_grad = False
            # self.weights.grad = None
            self.weights = None
        elif self.use_banded_matrix:
            # weight_h, weight_w = img_height, img_width
            weight_h, weight_w = img_height // self.weight_scale_factor, img_width // self.weight_scale_factor

            
            b_width = 2 * self.banded_matrix_width + 1
            self.weights = nn.Parameter(
                    torch.ones(num_psfs, weight_h * weight_w, b_width, device=self.device, requires_grad=True)
                )
        else:
            weight_h, weight_w = img_height // self.weight_scale_factor, img_width // self.weight_scale_factor
            print("weight slearning ", weight_h, weight_w)
            
            if self.weight_low_rank_factors  > 0:
                self.weights = nn.ParameterList([
                    nn.Parameter(torch.ones(num_psfs - 1, weight_h, self.weight_low_rank_factors , device=self.device, requires_grad=True)),
                    nn.Parameter(torch.ones(num_psfs - 1, self.weight_low_rank_factors, weight_w , device=self.device, requires_grad=True)),
                ])
            else:
                
                self.weights = nn.Parameter(
                    torch.ones(num_psfs - 1, weight_h, weight_w, device=self.device, requires_grad=True)
                )
            # if not learn_w:
            #     self.initialize_weights()

        # Initialize parameter values
        self._initialize_parameters(alpha=0.1)
        
    def eval(self):
        self.eval_mode = True 
        
    def train(self, mode = True):
        self.eval_mode = False
    
    def adjoint_check(self):
        x = torch.rand(16,3,self.img_height,self.img_width).to(self.device)
        
        Ax = self.forward(x)
        y = torch.rand_like(Ax)
        
        AT_y = self.adjoint(y) 
        
        v1 = torch.sum(Ax * y)
        v2 = torch.sum(x * AT_y)
        error = torch.abs(v1 - v2) / torch.max(torch.abs(v1), torch.abs(v2))
        
        print("ERROR ", v1 - v2 )
        
        print("v1 (Ax,y): ", v1.item())
        print("v2 (x,A*y): ", v2.item())
        print("Absolute difference: ", (v1 - v2).item())
        print("Relative error: ", error.item())
        print("Shapes - x:", x.shape, "Ax:", Ax.shape, "AT_y:", AT_y.shape)
        assert error < 1e-5, f'"A.T" is not the adjoint of "A". Check the definitions of these operators. Error: {error}'

    def weight_norm(self, p = 1):
        if self.weights is None:
            return 0
        
        weight_norm = self.weights.abs().norm(p=p, dim=-1).mean()
        weights_sum = self.weights.sum(axis=0).sum(axis=1).mean()
        
        return weight_norm + torch.abs(weights_sum - 1.0)

    def filter_norm(self, p=1):
        if self.filters is None:
            return 0 
        
        return torch.norm(self.filters, p=p, dim=0).mean() 
    
    def initialize_weights(self):
        # Create linearly interpolated weights for weights[0]
        linear_weights = torch.linspace(1, 0, self.weights.shape[1], device=self.device).view(-1, 1)
        linear_weights = linear_weights.expand(-1, self.weights.shape[2])
        
        # Assign weights[0]
        self.weights.data[0] = linear_weights

        # Assign weights[1] as 1 - weights[0]
        # self.weights.data[1] = 1 - linear_weights

        # Normalize weights along the 0th dimension (if needed)
        # self.weights.data /= self.weights.data.sum(dim=0, keepdim=True)
        
    def _initialize_parameters(self, alpha):
        
        # Initialize filters with normalized values
        if self.filters is not None:
            if self.kernel_init_func == 'random':
                with torch.no_grad():
                    init.uniform_(self.filters)
                    self.filters.data /= self.filters.sum(axis=[1,2,3], keepdims=True)
            elif self.kernel_init_func == 'delta':
                self.filters.data[:,0,self.kernel_size//2,self.kernel_size//2] = 1.
            else:
                raise ValueError(f"Unknown kernel init function : {self.kernel_init_func}")
        

        # Initialize weights with normalized values
        if self.num_psfs >= 2 and self.learn_w:
            if self.weight_low_rank_factors > 0:
                self.weights[0].data /= self.num_psfs
            else:
                if self.weight_init_func == 'unifom':
                    init.uniform_(self.weights)
                elif self.weight_init_func == 'constant':   
                    self.weights.data /= self.num_psfs  
                elif self.weight_init_func == 'random':   
                    print("***********************************     Random weights ")
                    self.weights.data = torch.rand_like(self.weights)
                    self.weights.data /= self.weights.data.sum(dim=2, keepdim=True)  # normalize across b_width
                    self.weights.data /= (self.num_psfs * 2)  # scale down
                else:
                    raise ValueError(f"Unknown weight init function : {self.weight_init_func}")
    
    def forward(self, x, weights = None, filters = None):
        batch_size, c, h, w = x.shape
        pad_size = self.kernel_size // 2
        num_psfs = self.num_psfs
        
        # weights = self.weight_enforcer(self.weights) 
        
        
        if filters is None:
            if self.non_blind:
                filters = self.true_filters[:self.num_psfs].unsqueeze(0) 
                # weights = weights[:,:,:self.num_psfs]
                # num_psfs = filters.shape[1]
            else:
                filters = self.filters
        
                
        # weighted_input = x.unsqueeze(2) * weights  # (batch_size, c, num_psfs, h, w)
        # weighted_input = weighted_input.reshape(batch_size * c , self.num_psfs, h, w)
        
        weighted_input = self.weight_enforcer(self.weights if weights is None else weights, x) 
        
        _, _, f_h, f_w = filters.shape
        self.num_psfs
        filters = filters.reshape(num_psfs, 1, f_h, f_w)
        
        if self.conv_fft:
            padded_weighted_input = F.pad(weighted_input, (pad_size, pad_size, pad_size, pad_size), mode='constant')
            
            filter_pad_left = (h + 2 * pad_size - f_h) // 2
            filter_pad_right = (h + 2 * pad_size - f_h) - filter_pad_left
            filter_pad_top = (h + 2 * pad_size - f_w) // 2
            filter_pad_bottom = (h + 2 * pad_size - f_w) - filter_pad_top
            filters = filters.reshape(num_psfs, 1, f_h, f_w)
            psfs_padded = F.pad(filters, (filter_pad_left, filter_pad_right, filter_pad_top, filter_pad_bottom))
            
            weighted_imgs_fft = torch.fft.fftn(padded_weighted_input, dim = (-2,-1))
            psfs_fft = torch.fft.fftn(torch.fft.ifftshift(psfs_padded, dim = (-2,-1)), dim = (-2,-1)).squeeze().unsqueeze(0)
            
            conv_out_res = torch.fft.ifftn(weighted_imgs_fft * psfs_fft, dim=(-2, -1)).real

            add_h, adj_w = conv_out_res.shape[-2:]  # Get height and width
            crop_top = pad_size
            crop_bottom = add_h - pad_size
            crop_left = pad_size
            crop_right = adj_w - pad_size
            
            conv_output = conv_out_res[:,:, crop_top : crop_bottom, crop_left : crop_right]
        else:                
            padded_weighted_input = F.pad(weighted_input, (pad_size, pad_size, pad_size, pad_size), mode='constant')
            conv_output = F.conv2d(padded_weighted_input, filters, groups=self.num_psfs)

        conv_output = conv_output.view(batch_size, c, self.num_psfs, h, w).sum(dim=2)  # Sum over num_psfs

        # print("Min max of conv output ", conv_output.min().item(), conv_output.max().item())
        return conv_output
    
    def adjoint(self, y):
        pad_size = self.kernel_size // 2
        
        if self.eval_mode and not self.non_blind:
            with torch.enable_grad():      
                x = torch.ones_like(y.detach()).requires_grad_()
                
                # Assume x and y have same shape.
                weights = self.weight_enforcer(self.weights)#.detach()#.requires_grad_(True)
                filters = self.filters.detach()#.requires_grad_(True)
            
                f = torch.sum(y.detach() * self.forward(x, weights = weights, filters = filters))
                # f = torch.sum(y.detach() * x)
                
                # print("After calling torch.autograd.grad(")
                adjoint_output = torch.autograd.grad(f, x, create_graph = True)[0]


                return adjoint_output
        
        '''
            # Manual way to compute adjoints during training
        '''
        batch_size, c, h, w = y.shape
        pad_size = self.kernel_size // 2

        y_expanded = y.unsqueeze(2).expand(-1, -1, self.num_psfs, -1, -1)  # (batch_size, c, num_psfs, h, w)

        y_weighted = y_expanded.view(batch_size * c, self.num_psfs, h, w)  # Reshape for convolution

        num_psfs = self.num_psfs
        
        if self.non_blind:
            filters = self.true_filters[:self.num_psfs].unsqueeze(0) 
            # num_psfs = filters.shape[1]
        else:
            filters = self.filters
                
        _, _, f_h, f_w = filters.shape
        
        if self.conv_fft:
            padded_weighted_input = F.pad(y_weighted, (pad_size, pad_size, pad_size, pad_size), mode='constant')
            
            filter_pad_left = (h + 2 * pad_size - f_h) // 2
            filter_pad_right = (h + 2 * pad_size - f_h) - filter_pad_left
            filter_pad_top = (h + 2 * pad_size - f_w) // 2
            filter_pad_bottom = (h + 2 * pad_size - f_w) - filter_pad_top
            filters = filters.reshape(num_psfs, 1, f_h, f_w)
            psfs_padded = F.pad(filters, (filter_pad_left, filter_pad_right, filter_pad_top, filter_pad_bottom))
            
            conv_output_adj = torch.fft.fftn(padded_weighted_input, dim=(-2, -1))  # FFT of the output (adjoint of IFFTN)
            psfs_fft = torch.fft.fftn(torch.fft.fftshift(psfs_padded, dim = (-2,-1)), dim = (-2,-1)).squeeze().unsqueeze(0).conj()

            # print("conv ", conv_output_adj.shape,  psfs_fft.shape)
            conv_transpose_output = conv_output_adj * psfs_fft

            conv_transpose_output = torch.fft.ifftn(conv_transpose_output, dim=(-2, -1)).real

            add_h, adj_w = conv_transpose_output.shape[-2:]  # Get height and width
            crop_top = pad_size
            crop_bottom = add_h - pad_size
            crop_left = pad_size
            crop_right = adj_w - pad_size

            conv_transpose_output = conv_transpose_output[:, :, crop_top:crop_bottom, crop_left:crop_right]
        else:
            filters_flipped = torch.flip(filters, dims=[-2, -1])  # Flip spatial dimensions
            filters_flipped = filters_flipped.reshape(num_psfs, 1, f_h, f_w)

            padded_weighted_input = F.pad(y_weighted, (pad_size, pad_size, pad_size, pad_size), mode='constant')
            conv_transpose_output = F.conv2d(padded_weighted_input, filters_flipped, groups=self.num_psfs)
            # conv_transpose_output = conv_transpose_output[...,]

        if not self.use_banded_matrix:
            weighted_input = conv_transpose_output.view(batch_size, c, self.num_psfs, h, w)
            weights = self.weight_enforcer(self.weights) 
            
            if self.non_blind:
                weights = weights[:,:,:self.num_psfs]

            adjoint_output = (weighted_input * weights).sum(dim=2)  # Sum over num_psfs
        else:
            b_w = 2 * self.banded_matrix_width + 1
            weighted_input = conv_transpose_output.view(batch_size * c, self.num_psfs, h * w)

            weights = self.weights

            if self.weight_scale_factor > 1:
                d = h // self.weight_scale_factor
                weight = weights.reshape(weights.shape[0],d,d,b_w).permute(0, 3, 1, 2)
                dw = self.weight_scale_factor * d
                weight = F.interpolate(weight, size=(dw, dw), mode='bilinear', align_corners=False).permute(0, 2, 3, 1)
                weights = weight.reshape(weight.shape[0], -1 , b_w)

            y_unfold = torch.einsum('bnp,npk->bpk', weighted_input, weights)  # (b*c, h*w, b_w)

            y_unfold = y_unfold.permute(0, 2, 1)  # (b*c, b_w, h*w)
            y_unfold = y_unfold.view(batch_size, c * b_w, h * w)

            pad = b_w // 2  # same as in forward
            padded_w = w + 2 * pad
            y_unfold = F.fold(y_unfold, output_size=(h, padded_w), kernel_size=(1, b_w), stride=1)

            adjoint_output = y_unfold[:, :, :, pad:pad + w]  # crop width dimension
        
        # print("Adjoint output  ", adjoint_output.min().item(), adjoint_output.max().item())
        
        return adjoint_output
    

if __name__ == "__main__":    
    # Initialize models
    # model_v1 = ShiftVaryingSystemCNNV1(512, 512, 2, 7).cuda()
    model_v2 = ShiftVaryingSystemCNN(512, 512, 2, kernel_size=5, banded_matrix_width = 5).cuda()
    model_v2.train()

    # # Synchronize parameters
    # # model_v1.filters = model_v2.filters
    # # model_v1.weights = model_v2.weights

    # # Generate input tensors
    input_tensor = torch.arange(16 * 3 * 512 * 512).reshape(16, 3, 512, 512).cuda().float()
    input_tensor /= input_tensor.numel()
    
    # start_time_v1 = time.time()
    # output_v1 = model_v1(input_tensor)
    # time_v1 = time.time() - start_time_v1

    # # Measure execution time for model_v2
    # start_time_v2 = time.time()
    output_v2 = model_v2(input_tensor)
    print("output v2 shape ", output_v2.shape, " in put  ", input_tensor.shape)
    model_v2.adjoint_check()
    print("pased adj check ")
    # time_v2 = time.time() - start_time_v2

    # # Compare outputs and display execution times
    # print(
    #     f"Shapes: {output_v1.shape}, {output_v2.shape}\n"
    #     f"All Equal: {torch.all(output_v1 == output_v2).item()}\n"
    #     f"Max Difference: {torch.max(output_v1 - output_v2).item()}\n"
    #     f"Time (model_v1): {time_v1:.4f}s, Time (model_v2): {time_v2:.4f}s"
    # )
    # adjoint = model_v2.adjoint(output_v2)

    # print("Adjoint output.shape ", adjoint.shape)
    # print("Adjoint check ", model_v2.adjoint_check())
    
    
    measurement_operator = ShiftVaryingBlur(kernel_size=5, device='cpu', kernel_type="motion", alpha = 0.75)
    # print("Adjoint check ", measurement_operator.adjoint_check())
    
    model_v2.initialize_weights()
    model_v2.filters.data = torch.stack([
        measurement_operator.get_kernel(measurement_operator.kernel_size, alpha = 0), 
        measurement_operator.get_kernel(measurement_operator.kernel_size, alpha = 1)
        ], dim = 0).to(model_v2.device).unsqueeze(1)
    print("model_v2.filters. ", model_v2.filters.shape)
    out_meas = measurement_operator(input_tensor)
    out_model_v2 = model_v2(input_tensor)
    
    print("outp means ", torch.max(out_meas), torch.max(out_model_v2))
    
    print(" equal ",torch.max(out_meas - out_model_v2), out_meas[0,0,0,0].item() , out_model_v2[0,0,0, 0].item(), input_tensor[0,0,:3,0].sum().item()/5)

    # from models.sv_miniscope import SVMiniscope
    # from utils import gen_patterns

    # miniscope = ShiftVaryingBlur(15, 'cuda', 'MLA')
    # x = gen_patterns.generate_dot_grid()[:,:,None]
    # x = x.astype(np.float32) / 255.

    # x = torch.from_numpy(x).permute(2,0,1).unsqueeze(0).cuda()

    # x = F.interpolate(x, size=(256, 256), mode='bilinear', align_corners=False)
    # y = miniscope(x)
    # Aty = miniscope.adjoint(y)

    # sv_model = ShiftVaryingSystemCNN(256, 256, num_psfs = 32, kernel_size = 241, weight_scale_factor = 4)
    # sv_model.eval()
    
    # from measurements import ShiftVaryingBlur
    # meas = ShiftVaryingBlur(15, 'cuda', 'MLA')
    
    # # sv_model.true_filters = meas.get_true_filters()
    # # sv_model.weights.data = meas.get_weights_filters()
    
    # y = meas(x)
    # from fista import FISTA
    # fista = FISTA( max_iter=1 )
    # x_hat, _, _ = fista.solve(y, operator=sv_model, step_size = 0.1, debug = True, x_true = x)
    # x_hat, _, _ = fista.solve(y, operator=meas, step_size = 0.1, debug = True, x_true = x)
    
    
        
    # fig, axes = plt.subplots(1, 4, figsize=(12, 4))

    # x_np = x.squeeze().detach().cpu().numpy()
    # y_np = y.squeeze().detach().cpu().numpy()
    # Aty_np = Aty.squeeze().detach().cpu().numpy()
    # # Plot each tensor
    # titles = ['x', 'y', 'Aty']
    # for ax, img, title in zip(axes, [x_np, y_np, Aty_np], titles):
    #     ax.imshow(img, cmap='gray')
    #     ax.set_title(title)
    #     ax.axis('off')
    
    # axes[-1].imshow(x_hat.squeeze().detach().cpu().numpy(), cmap='gray')
    # axes[-1].set_title("xhat")
    # axes[-1].axis('off')
    # # Save to file
    # plt.savefig('zz_output_SVMODEL.png', bbox_inches='tight', dpi=300)
    # plt.show()


    # sv_model.adjoint_check()
    # print("pased adj check ")
    # print("wordard error: ", meas(x) - sv_model(x))