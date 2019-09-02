from graphviz import Digraph
import torch
from torch.autograd import Variable
from torchviz import make_dot, make_dot_from_trace


def show_model2(model):
    """
    second way to show model by using Hiddenlayer module
    link: https://github.com/waleedka/hiddenlayer
    there is a warning, just ignore it
    :param module:
    :param model_name:
    :return:
    """
    try:
        import hiddenlayer as hl
    except ImportError:
        import os
        os.system('pip install hiddenlayer')
        import hiddenlayer as hl

    inp1 = torch.randn(1, 3, 300, 300)
    inp2 = torch.randn(1, 1, 300, 300)
    inputs = (Variable(inp1), Variable(inp2))
    # if model_name=='PSPNet':
    #     model = getattr(module, model_name)(sizes=(1, 2, 3, 6),
    #                                         psp_size=512, deep_features_size=256, backend='resnet18',pretrained=False)
    # else:
    #     model = getattr(module, model_name)(pretrained=False)
    print(model)
    g = hl.build_graph(model,inputs)
    g.save('./model_graph.pdf')


def show_model3(model):
    """
    use the torchviz package to show the graph
    :param module:
    :param model_name:
    :return:
    """
    inp1 = torch.randn(1, 3, 300, 300)
    inp2 = torch.randn(1, 3, 300, 300)
    inputs = (Variable(inp1), Variable(inp2))

    print(model)
    g = make_dot(model(inputs), params=dict(model.named_parameters()))
    g.view()


if __name__ =='__main__':

    pass
