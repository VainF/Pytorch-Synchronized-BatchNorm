import torch
from torch.autograd import Function
from torch.autograd.function import once_differentiable
import torch.nn.functional as F

def unsqueeze(tensor):
    return tensor.unsqueeze(1).unsqueeze(0)
    
class SyncBNFunction(Function):

    @staticmethod
    def forward(ctx, x, weight, bias, running_mean, running_var, momentum, eps, training, bn_ctx):
        x_shape = x.shape
        B, C = x_shape[:2]

        _x = x.view(B,C,-1).contiguous()

        ctx.eps = eps
        ctx.training = training

        ctx.sync = bn_ctx.sync
        ctx.cur_device = bn_ctx.cur_device
        ctx.queue = bn_ctx.queue
        ctx.is_master = bn_ctx.is_master
        ctx.devices = bn_ctx.devices

        norm = 1/(_x.shape[0] * _x.shape[2])

        if ctx.training:
            _ex = _x.sum(2).sum(0) * norm
            _exs = _x.pow(2).sum(2).sum(0) * norm

            if ctx.sync:
                if ctx.is_master:

                    _ex, _exs = [_ex.unsqueeze(1)], [_exs.unsqueeze(1)]

                    master_queue = ctx.queue[0]
                    for j in range(master_queue.maxsize):
                        _slave_ex, _slave_exs = master_queue.get()
                        master_queue.task_done()

                        _ex.append(  _slave_ex.unsqueeze(1)  )
                        _exs.append( _slave_exs.unsqueeze(1) )
                    
                    _ex  = torch.cuda.comm.gather( _ex,  dim=1 ).mean(1)
                    _exs = torch.cuda.comm.gather( _exs, dim=1 ).mean(1)

                    distributed_tensor = torch.cuda.comm.broadcast_coalesced( (_ex, _exs), ctx.devices )

                    for dt, q in zip( distributed_tensor[1:], ctx.queue[1:] ):
                        q.put(dt)
                else:
                    master_queue = ctx.queue[0]
                    slave_queue = ctx.queue[ctx.cur_device]
                    master_queue.put( (_ex, _exs) )

                    _ex, _exs = slave_queue.get()
                    slave_queue.task_done()
                    _ex, _exs = _ex.squeeze(), _exs.squeeze()
            
            _var = _exs - _ex.pow(2)
            N = B*len(ctx.devices)
            unbiased_var = _var * N / (N - 1)
            
            running_mean.mul_( (1-momentum) ).add_( momentum * _ex  )
            running_var.mul_( (1-momentum) ).add_( momentum * unbiased_var )
            ctx.mark_dirty(running_mean, running_var)
        else:
            _ex, _var = running_mean.contiguous(), running_var.contiguous()
            _exs = _ex.pow(2) + _var
            
        invstd = 1/torch.sqrt( _var + eps )

        if weight is not None: # affine
            output = (_x - unsqueeze(_ex) ) * unsqueeze(invstd) * unsqueeze(weight)  + unsqueeze(bias)
        else:
            output = (_x - unsqueeze(_ex) ) * unsqueeze(invstd)
        
        ctx.save_for_backward(x, _ex, _exs, weight, bias)
        return output.view(*x_shape).contiguous().clone()

    @staticmethod
    def backward(ctx, grad_output):
        x, _ex, _exs, weight, bias = ctx.saved_tensors
        grad_x = grad_weight = grad_bias = None

        B,C = grad_output.shape[:2]
        grad_output_shape = grad_output.shape

        _var = _exs - _ex.pow(2)
        _std = torch.sqrt( _var + ctx.eps)
        invstd = 1.0 / _std

        grad_output = grad_output.view(B,C,-1)
        x = x.view(B,C,-1)

        norm = 1.0/(x.shape[0] * x.shape[2])
        
        dot_p = ( grad_output * ( x -  unsqueeze( _ex ) ) ).sum(2).sum(0)
        grad_output_sum = grad_output.sum(2).sum(0)

        grad_scale = weight * invstd

        grad_ex  = -grad_output_sum * grad_scale + _ex * invstd * invstd * dot_p * grad_scale
        grad_exs = -0.5 * grad_scale * invstd * invstd * dot_p 

        # Sync
        if ctx.training:
            if ctx.sync:
                if ctx.is_master: 
                    grad_ex, grad_exs = [grad_ex.unsqueeze(1)], [grad_exs.unsqueeze(1)]
                    master_queue = ctx.queue[0]
                    for j in range(master_queue.maxsize):
                        grad_slave_ex, grad_slave_exs = master_queue.get()
                        master_queue.task_done()

                        grad_ex.append(  grad_slave_ex.unsqueeze(1)  )
                        grad_exs.append( grad_slave_exs.unsqueeze(1) )

                    grad_ex  = torch.cuda.comm.gather( grad_ex,  dim=1 ).mean(1)
                    grad_exs = torch.cuda.comm.gather( grad_exs, dim=1).mean(1)

                    distributed_tensor = torch.cuda.comm.broadcast_coalesced( (grad_ex, grad_exs), ctx.devices )
                    for dt, q in zip( distributed_tensor[1:], ctx.queue[1:] ):
                        q.put(dt)
                else:
                    master_queue = ctx.queue[0]
                    slave_queue = ctx.queue[ctx.cur_device]
                    master_queue.put( (grad_ex, grad_exs) )

                    grad_ex, grad_exs = slave_queue.get()
                    slave_queue.task_done()
                    grad_ex, grad_exs = grad_ex.squeeze(), grad_exs.squeeze()
        
        if ctx.needs_input_grad[0]:
            grad_x = grad_output * unsqueeze( grad_scale ) + unsqueeze( grad_ex * norm ) +  unsqueeze(grad_exs) * 2 * x * norm 

        if ctx.needs_input_grad[1]:
            grad_weight = dot_p * invstd
        
        if ctx.needs_input_grad[2]:
            grad_bias = grad_output_sum
        
        return grad_x.view(*grad_output_shape), grad_weight, grad_bias, None, None, None, None, None, None

        

        








    
