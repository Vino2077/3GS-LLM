"""Check prefix gradients, masking, echo discrimination and strict judge validation."""
import torch
from train import prefix_loss_weights
from stage3_quality import copy_metrics, prompt_rejection, near_eval, normalized
from teacher_client import acceptable, validate_judgment
from conditional_loss import mismatched_batch, response_losses


def main():
    mask = torch.tensor([[0, 0, 1, 1, 1, 1, 0], [0, 1, 1, 0, 0, 0, 0]])
    targets = torch.tensor([[1, 4, 55, 77, 88, 2, 0], [4, 33, 2, 0, 0, 0, 0]])
    weighted = prefix_loss_weights(mask, targets, 2, 1.5)
    expected = torch.tensor([[0, 0, 1.5, 1.5, 1, 1, 0], [0, 1.5, 1, 0, 0, 0, 0]])
    torch.testing.assert_close(weighted, expected)
    losses = torch.ones_like(weighted, requires_grad=True)
    loss = (losses * weighted).sum() / weighted.sum()
    loss.backward()
    torch.testing.assert_close(losses.grad, expected / expected.sum())
    assert not torch.any(losses.grad[mask == 0])
    x = torch.tensor([[1, 3, 21, 4, 61, 62], [1, 3, 22, 4, 71, 72]])
    y = torch.tensor([[3, 21, 4, 61, 62, 2], [3, 22, 4, 71, 72, 2]])
    m = torch.tensor([[0, 0, 0, 1, 1, 1], [0, 0, 0, 1, 1, 1]])
    nx, ny, nm, usable = mismatched_batch(x, y, m)
    assert nx[0].tolist() == [1, 3, 22, 4, 61, 62]
    assert ny[0][nm[0].bool()].tolist() == [61, 62, 2]
    assert usable.all()
    logits = torch.randn(2, 6, 80, requires_grad=True)
    response_losses(logits, y, m).sum().backward()
    assert not logits.grad[m == 0].any()
    assert not logits.grad[y == 2].any()
    assert copy_metrics("Скайрим вечен?", "Скайрим вечен? А что не так?")["prompt_copy"]
    assert not copy_metrics("Скайрим вечен?", "Скайрим ещё переживёт нас благодаря модам.")["prompt_copy"]
    assert copy_metrics("Как дела?", "Как дела?")["exact_echo"]
    assert prompt_rejection("А слева?") is not None
    assert prompt_rejection("Ну выше же написали") is not None
    assert prompt_rejection("Скайрим вечен?") is None
    assert near_eval("СКАЙРИМ вечен!", [normalized("Скайрим вечен?")])
    assert not near_eval("Люблю сыр", [normalized("Скайрим вечен?")])
    good = dict(relevance=5, directness=5, fluency=5, style=4, non_repetition=5,
                hidden_context=False, prompt_copy=False)
    validate_judgment({"self_contained": True, "scores": [good]}, 1)
    assert acceptable(good)
    assert not acceptable({**good, "relevance": 3})
    assert not acceptable({**good, "prompt_copy": True})
    try:
        validate_judgment({"self_contained": True, "scores": [{**good, "relevance": True}]}, 1)
    except ValueError:
        pass
    else:
        raise AssertionError("bool score accepted as int")
    print("PASS: prefix weighting, masked gradients, EOS, padding, copying, exclusion, judge schema")


if __name__ == "__main__":
    main()
