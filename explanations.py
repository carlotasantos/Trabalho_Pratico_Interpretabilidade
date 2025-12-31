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


def occlusion_map(model, x, target, patch=4):
    model.eval()

    with torch.no_grad():
        base_score = model(x)[0, target].item()

    _, _, H, W = x.shape
    heatmap = torch.zeros((H, W), device=x.device)


    for i in range(0, H, patch):
        for j in range(0, W, patch):
            x_occ = x.clone()
            x_occ[:, :, i:i+patch, j:j+patch] = 0.0  # tapa a região

            with torch.no_grad():
                score = model(x_occ)[0, target].item()


            importance = base_score - score

            heatmap[i:i+patch, j:j+patch] = importance

    return heatmap.abs().cpu().numpy()


def guided_backprop(model, x, target):
    hooks = []

    def hook_fn(module, grad_in, grad_out):
        return (torch.clamp(grad_in[0], min=0.0),)

    for m in model.modules():
        if isinstance(m, torch.nn.ReLU):
            hooks.append(m.register_full_backward_hook(hook_fn))

    x = x.clone().detach().requires_grad_(True)

    out = model(x)
    score = out[0, target]

    model.zero_grad()
    score.backward()

    for h in hooks:
        h.remove()

    return x.grad.abs().squeeze().cpu().numpy()

def grad_cam(model, x, target):
    import torch.nn.functional as F

    acts = None
    grads = None

    def save_acts(module, inp, out):
        nonlocal acts
        acts = out

    def save_grads(module, grad_in, grad_out):
        nonlocal grads
        grads = grad_out[0]

    h1 = model.conv2.register_forward_hook(save_acts)
    h2 = model.conv2.register_full_backward_hook(save_grads)

    out = model(x)
    score = out[0, target]

    model.zero_grad()
    score.backward()

    h1.remove()
    h2.remove()

    # pesos
    w = grads.mean(dim=(2, 3), keepdim=True)          # (1, C, 1, 1)
    cam = (w * acts).sum(dim=1, keepdim=True)         # (1, 1, h, w)
    cam = F.relu(cam)

    cam = F.interpolate(cam, size=(28, 28), mode="bilinear", align_corners=False)

    return cam.squeeze().detach().cpu().numpy()
