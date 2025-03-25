import torch


def scaling_hands(hands, meta):
    # pred_hands: [B, 2, T, 2]
    # h, w: [B]
    B, _, T, _ = hands.shape
    h, w = meta[2:4]
    h = h[:, None, None, None].cuda()
    w = w[:, None, None, None].cuda()
    mask = torch.cat((torch.zeros((B, 2, T, 1)), torch.ones((B, 2, T, 1))), dim=3).cuda()
    hands = hands * (h * mask + w * (1 - mask))
    return hands


def scaling_original(preds, meta):
    h, w = meta[2:4]
    h = h[:, None].cuda()
    w = w[:, None].cuda()
    odd = torch.zeros(preds.shape).cuda()
    even = torch.zeros(preds.shape).cuda()
    for i in range(16):
        if i % 2:
            odd[:, i] = 1
        else:
            even[:, i] = 1
    preds = preds * (h * odd + w * even)
    return preds
