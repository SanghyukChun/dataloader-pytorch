import tqdm


def evaluate_from_dataloader(model, dataloader, verbose=True, use_tqdm=False,
                             cuda=False, phase='test', print_fn=print):
    n_correct, n_total = 0, 0
    if cuda:
        model = model.cuda()
    model.eval()
    wrapper = tqdm.tqdm if use_tqdm else lambda x: x
    for x, label in wrapper(dataloader):
        if cuda:
            x = x.cuda()
            label = label.cuda()
        preds = model(x)
        preds = preds.data.max(1)[1]
        n_correct += int(sum(label == preds))
        n_total += len(label)

    acc = n_correct / float(n_total)
    if verbose:
        print_fn('==== {} accuraccy: {}'.format(phase, acc))
    return acc
