"""Optional small pair-ranking loss. No model/runtime parameters are added."""
import torch
import torch.nn.functional as F


def mismatched_batch(inputs, targets, mask, context=256):
    """Keep complete responses but rotate their user prefixes within a batch."""
    n = inputs.shape[0]
    starts = mask.bool().long().argmax(dim=1).tolist()
    rows, lengths = [], []
    usable = []
    for i in range(n):
        j = (i + 1) % n
        answer = targets[i][mask[i].bool()]
        other_answer = targets[j][mask[j].bool()]
        prefix = inputs[j, :starts[j] + 1]
        original_prefix = inputs[i, :starts[i] + 1]
        room = context - len(answer) - 3
        if room < 0:
            raise ValueError('Response alone exceeds context')
        user = prefix[2:-1]
        if len(user) > room:
            user = user[-room:] if room else user[:0]
        prefix = torch.cat((prefix[:2], user, prefix[-1:]))
        rows.append(torch.cat((prefix, answer)))
        lengths.append(len(prefix))
        usable.append(not torch.equal(original_prefix, prefix) and not torch.equal(answer, other_answer))
    width = max(len(row) for row in rows)
    tokens = inputs.new_zeros((n, width))
    masks = mask.new_zeros((n, width - 1))
    for i, row in enumerate(rows):
        tokens[i, :len(row)] = row
        masks[i, lengths[i] - 1:len(row) - 1] = 1
    return tokens[:, :-1], tokens[:, 1:], masks, torch.tensor(usable, device=inputs.device)


def response_losses(logits, targets, mask, prefix=16):
    weights = mask.float() * (mask.cumsum(dim=1) <= prefix) * (targets != 2)
    losses = F.cross_entropy(logits.float().transpose(1, 2), targets, reduction='none')
    return (losses * weights).sum(dim=1) / weights.sum(dim=1).clamp_min(1)


def ranking_loss(model, logits, inputs, targets, mask, margin=.25):
    nx, ny, nm, usable = mismatched_batch(inputs, targets, mask, model.config.context_length)
    negative_logits, _ = model(nx)
    positive = response_losses(logits, targets, mask)
    negative = response_losses(negative_logits, ny, nm)
    values = F.relu(margin + positive - negative)
    return (values * usable).sum() / usable.sum().clamp_min(1)
