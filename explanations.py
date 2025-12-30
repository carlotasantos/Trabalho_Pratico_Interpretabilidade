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
