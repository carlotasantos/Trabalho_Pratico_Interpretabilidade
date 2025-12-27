import torch

def saliency_gradient(model, x, target):
    x = x.clone().detach().requires_grad_(True)

    logits = model(x)
    score = logits[0, target]

    model.zero_grad()
    score.backward()

    return x.grad.abs().squeeze().detach().cpu().numpy()


def integrated_gradients(model, x, target, steps=20):
    baseline = torch.zeros_like(x)
    grads_sum = torch.zeros_like(x)

    for k in range(1, steps + 1):
        alpha = k / steps
        x_int = baseline + alpha * (x - baseline)
        x_int.requires_grad_()

        out = model(x_int)
        score = out[0, target]

        grad = torch.autograd.grad(score, x_int)[0]
        grads_sum += grad

    ig = (x - baseline) * (grads_sum / steps)
    return ig.abs().squeeze().detach().cpu().numpy()
