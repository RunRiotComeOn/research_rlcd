"""Constrained Action plus a three-digit integer percentage (000% through 100%)."""
import torch

from constrained_policy import action_prefix, logits_for, token_ids


def percent_prompt(row):
    from core import prompt_text
    return prompt_text(row).replace('Confidence: <0-9>', 'Confidence: <000-100>%')


def digit_ids(processor):
    return token_ids(processor, 2)[1]


def percent_prefix(processor, action, digits=''):
    prefix = processor.tokenizer.encode(
        f'Action: {chr(65+action)}\nConfidence: ', add_special_tokens=False)
    ids = digit_ids(processor)
    return prefix + [ids[int(d)] for d in digits]


def distribution(model, processor, batch, action, digits, candidate_digits):
    ids = digit_ids(processor)
    return logits_for(model, batch, percent_prefix(processor, action, digits),
                      [ids[d] for d in candidate_digits]).softmax(-1)


def sample_percent(model, processor, batch, action):
    """Return a percentage and log-probability support path."""
    first = distribution(model, processor, batch, action, '', [0, 1])
    d0 = int(torch.multinomial(first, 1))
    if d0 == 1:
        return 100
    tens = distribution(model, processor, batch, action, '0', list(range(10)))
    d1 = int(torch.multinomial(tens, 1))
    units = distribution(model, processor, batch, action, f'0{d1}', list(range(10)))
    d2 = int(torch.multinomial(units, 1))
    return 10*d1+d2


def percent_logprob(model, processor, batch, action, value):
    ids = digit_ids(processor)
    first = logits_for(model, batch, percent_prefix(processor, action), [ids[0], ids[1]]).log_softmax(-1)
    if value == 100:
        return first[1]
    tens, units = divmod(value, 10)
    second = logits_for(model, batch, percent_prefix(processor, action, '0'), ids).log_softmax(-1)
    third = logits_for(model, batch, percent_prefix(processor, action, f'0{tens}'), ids).log_softmax(-1)
    return first[0] + second[tens] + third[units]


def greedy_percent(model, processor, batch, num_choices):
    actions, _ = token_ids(processor, num_choices)
    with torch.no_grad():
        ap = logits_for(model, batch, action_prefix(processor), actions).softmax(-1)
        action = int(ap.argmax())
        first = distribution(model, processor, batch, action, '', [0, 1])
        # Greedy path, matching the existing digit-by-digit constrained decoder.
        if int(first.argmax()) == 1:
            value = 100
        else:
            tens = int(distribution(model, processor, batch, action, '0', list(range(10))).argmax())
            units = int(distribution(model, processor, batch, action, f'0{tens}', list(range(10))).argmax())
            value = 10*tens+units
    return {'output': f'Action: {chr(65+action)}\nConfidence: {value:03d}%',
            'action_index': action, 'percent': value, 'confidence': value/100,
            'action_probabilities': ap.tolist()}
