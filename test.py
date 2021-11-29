from model_paddle import DeeplabMulti

import paddle
import paddle.optimizer as optim
from paddle.optimizer.lr import PolynomialDecay

def lr_poly(base_lr, iter, max_iter, power):
    return base_lr*((1-float(iter)/max_iter)**(power))

def adjust_learning_rate(optimizer, i_iter):
    lr = lr_poly(args.learning_rate, i_iter, args.num_steps, args.power)
    optimizer.set_lr(lr)

if __name__ == '__main__':
    model=DeeplabMulti()
    learning_rate=PolynomialDecay(learning_rate=2.5e-4,decay_steps=150000,power=0.9)
    optimizer=optim.Momentum(learning_rate=learning_rate,parameters=model.parameters())
    # optimizer.clear_grad()
    # for i in range(5000):
    #     optimizer.clear_grad()
    #     lr=lr_poly(2.5e-4,i,150000,0.9)
    #     # print(f'iter: {i} lr:{optimizer.get_lr()}  {lr} {optimizer.get_lr()==lr}')
    #     optimizer.step()
    #     optimizer._learning_rate.step()
    print(optimizer.state_dict())
    state=paddle.load('optimizer.pdparams')
    print(state)
    optimizer.set_state_dict(state)
    print(optimizer.state_dict())
    print(optimizer._learning_rate.last_epoch)
    print(optimizer.get_lr())