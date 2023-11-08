from typing import Any
import torch
import torch.nn as nn
import torch.nn.utils.parametrize as parametrize
import copy
import torch.nn.functional as F

torch.manual_seed(0)

    


class WeightAverageParametrization(nn.Module):
    def __init__(self, w, alpha, name, property):
        super(WeightAverageParametrization, self).__init__()
        self.w = w
        self.alpha = alpha
        self.name = name
        self.property = property

    def forward(self, x):
        w_layer = []
        for w_model in self.w:
            layer = w_model[f"{self.name}.{self.property}"]
            w_layer.append(layer)
        # w_avg is a tensor
        w_avg = copy.deepcopy(w_layer[0])
        # initialize w_avg to 0'
        w_avg = torch.zeros(w_avg.shape, device=w_avg.device)

        for i in range(len(w_layer)):
            w_avg += w_layer[i] * self.alpha[i]
        w_avg = torch.div(w_avg, torch.sum(self.alpha))

        return w_avg


class WeightAverage(nn.Module):
    def __init__(self, w, alpha, property):
        super(WeightAverage, self).__init__()
        self.w = w
        self.alpha = alpha
        self.property = property

    def forward(self, x):
        w_avg = copy.deepcopy(self.w[0])
        for key in w_avg.keys():
            w_avg[key] = torch.zeros(w_avg[key].shape, device=w_avg[key].device)
        for key in w_avg.keys():
            for i in range(len(self.w)):
                w_avg[key] += self.w[i][key] * self.alpha[i]
            w_avg[key] = torch.div(w_avg[key], torch.sum(self.alpha))
        print("w_avg: ", w_avg)
        if(self.property == "weight"):
            return w_avg["weight"]
        elif(self.property == "bias"):
            return w_avg["bias"]


class SimpleMLP(nn.Module):
    def __init__(self):
        super().__init__()
        # self.hidden = nn.Linear(5, 5)
        # self.relu = nn.ReLU()
        self.output = nn.Linear(5, 5)


    def forward(self, X):
        # return self.output(self.relu(self.hidden(X)))
        return self.output(X)



# create 5 SimpleMLP models
layer1 = SimpleMLP()
# print("layer1 state_dict: ", layer1.state_dict())

# for module in layer1.modules():
#     print(module)

layer2 = SimpleMLP()
layer3 = SimpleMLP()
layer4 = SimpleMLP()
layer5 = SimpleMLP()

# freeze the parameters of the 5 input layers
for layer in [layer1, layer2, layer3, layer4, layer5]:
    for param in layer.parameters():
        param.requires_grad = False

# create another linear layer with the same input and output features as the input layers
layer6 = SimpleMLP()
# create a learnable parameter for the weight coefficient with 5 elements that sum up to 1
layer6.alpha = nn.Parameter(torch.ones(5) / 5)

# create an optimizer that will only update layer6.alpha
optimizer = torch.optim.Adam([layer6.alpha], lr=0.01)


# print("weight before: ", layer6.weight)
global_model = SimpleMLP()

print("global_model state_dict before: ", global_model.state_dict())


def model_register_parametrization(model, input_models, alpha):
    for name, module in model.named_modules():
        if(isinstance(module, nn.Linear)):
            parametrize.register_parametrization(module, "weight", WeightAverageParametrization(input_models, alpha, name, "weight"))
            parametrize.register_parametrization(module, "bias", WeightAverageParametrization(input_models, alpha, name, "bias"))

def model_remove_parametrizations(model):
    for name, module in model.named_modules():
        if(isinstance(module, nn.Linear)):
            parametrize.remove_parametrizations(module, "weight", leave_parametrized=True)
            parametrize.remove_parametrizations(module, "bias", leave_parametrized=True)

test = torch.randn(1, 5)
print("test: ", test)

for i in range(1):
    x = torch.randn(1, 5)
    # parametrize.register_parametrization(layer6, "weight", WeightAverage([layer1.state_dict(), layer2.state_dict(), layer3.state_dict(), layer4.state_dict(), layer5.state_dict()], layer6.alpha, "weight"))
    # parametrize.register_parametrization(layer6, "bias", WeightAverage([layer1.state_dict(), layer2.state_dict(), layer3.state_dict(), layer4.state_dict(), layer5.state_dict()], layer6.alpha, "bias"))

    model_register_parametrization(layer6, [layer1.state_dict(), layer2.state_dict(), layer3.state_dict(), layer4.state_dict(), layer5.state_dict()], layer6.alpha)

    out = layer6(x)
    out.mean().backward()

    print("\nlayer6 output weight before: ", layer6.output.weight)

    optimizer.step()

    print("\nlayer6 output weight after: ", layer6.output.weight)


    model_remove_parametrizations(layer6)

    # print("\nlayer6 output weight final: ", layer6.output.weight)

    # copy the parametrize weights to the global model
    global_model.load_state_dict(layer6.state_dict(), strict=False)

    print("\nglobal_model state_dict: ", global_model.state_dict())
    


# print("global_model", global_model.state_dict())

print("alpha: ", layer6.alpha)
print("layer6 output: ", layer6(test))
print("global model output: ", global_model(test))





