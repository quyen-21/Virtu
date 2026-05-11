def score_candidate_layout(items,rejected,metrics):
    s=float(metrics.get('fastScore',.5)); p=metrics.get('penalties',{})
    s-=.04*len(rejected)+.06*p.get('collision',0)+.08*p.get('unsupported',0)+.03*p.get('wall',0)
    if items: s+=.1*sum(float(x.get('keepProbability',.5)) for x in items)/len(items)
    return max(0,min(1,s))
def choose_best_layout(candidates):
    if not candidates: return {'items':[],'rejected':[],'metrics':{'layoutScore':0}}
    for c in candidates: c['metrics']['layoutScore']=score_candidate_layout(c.get('items',[]),c.get('rejected',[]),c.get('metrics',{}))
    return sorted(candidates,key=lambda c:c['metrics'].get('layoutScore',0),reverse=True)[0]
