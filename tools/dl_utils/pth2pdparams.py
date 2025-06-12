"""
PyTorch模型参数转PaddlePaddle模型参数工具

功能：
- 将PyTorch(pth)参数文件转换为PaddlePaddle(pdparams)格式。
- 支持参数名适配、模型参数结构对齐。

依赖：torch, paddle, os
"""
import torch
import paddle
import os

def adapt_param_name(key: str) -> str:
    """参数名适配（BN等）"""
    if 'running_mean' in key:
        key = key.replace('running_mean', '_mean')
    if 'running_var' in key:
        key = key.replace('running_var', '_variance')
    return key

def pth2pdparams(
    pth_path: str,
    paddle_model,
    save_path: str = ''
):
    """
    PyTorch参数转PaddlePaddle参数主函数

    :param pth_path: pth文件路径
    :param paddle_model: 已实例化的paddle模型
    :param save_path: pdparams保存路径（默认同名）
    :return: None
    """
    if not save_path:
        save_path = pth_path.replace('.pth', '.pdparams')
    state_pth = torch.load(pth_path, map_location='cpu')
    new_params = paddle_model.state_dict().copy()
    for i in state_pth:
        i_parts = i.split('.')
        if not i_parts[1] == 'layer5':
            new_key = '.'.join(i_parts[1:])
            new_params[new_key] = state_pth[i].cpu().detach().numpy()
    # 参数名适配
    state_pdparams = {adapt_param_name(k): v for k, v in new_params.items()}
    paddle_model.set_state_dict(state_pdparams)
    paddle.save(state_pdparams, save_path)
    print(f'权重已保存到: {save_path}')

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='PyTorch参数转PaddlePaddle参数工具')
    parser.add_argument('--pth', required=True, help='pth文件路径')
    parser.add_argument('--paddle-model-fn', required=True, help='Paddle模型构造函数（字符串）')
    parser.add_argument('--pdparams', default='', help='pdparams保存路径（可选）')
    parser.add_argument('--paddle-args', type=str, default='{}', help='Paddle模型参数，json格式')
    args = parser.parse_args()
    import json
    paddle_args = json.loads(args.paddle_args)
    paddle_model = eval(args.paddle_model_fn)(**paddle_args)
    pth2pdparams(
        pth_path=args.pth,
        paddle_model=paddle_model,
        save_path=args.pdparams
    )
