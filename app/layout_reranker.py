def score_candidate_layout(items, rejected, metrics):
    s = float(metrics.get('fastScore', .5))
    p = metrics.get('penalties', {})
    intent = str(metrics.get('designIntent', 'generic'))
    intent_bonus = {'balanced_symmetry': .08, 'cozy': .06, 'minimal': .06, 'workflow': .08}.get(intent, 0.0)
    s -= .04 * len(rejected) + .06 * p.get('collision', 0) + .08 * p.get('unsupported', 0) + .03 * p.get('wall', 0) + .05 * p.get('outside', 0) + .08 * p.get('door', 0)
    s += .18 * float(metrics.get('relationScore', .5)) + .18 * float(metrics.get('facingScore', .5)) + .14 * float(metrics.get('clearanceScore', .5)) + .14 * float(metrics.get('aestheticScore', .5))
    s += .06 * float(metrics.get('wallAlignmentScore', .5)) + .06 * float(metrics.get('symmetryScore', .5)) + .06 * float(metrics.get('roomBalanceScore', .5))
    s -= .08 * float(metrics.get('collisionCount', 0)) + .08 * float(metrics.get('outsideCount', 0)) + .06 * float(metrics.get('doorPenaltyCount', 0))
    s += intent_bonus
    s += .14 * float(metrics.get('cameraVisibilityScore', .5)) + .06 * float(metrics.get('visibilityFront', .5))
    if items:
        s += .1 * sum(float(x.get('keepProbability', .5)) for x in items) / len(items)
    return max(0, min(1, s))


def choose_best_layout(candidates):
    if not candidates:
        return {'items': [], 'rejected': [], 'metrics': {'layoutScore': 0}}
    valid = []
    for c in candidates:
        metrics = c['metrics']
        if metrics.get('hardRejected'):
            metrics['layoutScore'] = 0.0
        else:
            metrics['layoutScore'] = score_candidate_layout(c.get('items', []), c.get('rejected', []), metrics)
        valid.append(c)
    ranked = sorted(valid, key=lambda c: (c['metrics'].get('hardPass', True), c['metrics'].get('layoutScore', 0)), reverse=True)
    return ranked[0]
