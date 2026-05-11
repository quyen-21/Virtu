
import math
WALL_CATEGORIES={'wall_art','mirror','curtain','window','door'}
FLOOR_UNDER_CATEGORIES={'rug'}
SUPPORTER_CATEGORIES={'coffee_table','dining_table','desk','nightstand','cabinet','shelf','tv_stand','counter'}
NEEDS_SUPPORT_CATEGORIES={'tv','vase','book','decor','lamp','plant'}
def fp(i):
    f=i.get('footprint') or {}; return float(f.get('widthM',i.get('widthM',.5))), float(f.get('depthM',i.get('depthM',.5))), float(f.get('heightM',i.get('heightM',.5)))
def clamp(v,a,b): return max(a,min(b,v))
def snap_inside(i,room):
    rw=float(room.get('widthM',4)); rl=float(room.get('lengthM',5)); w,d,h=fp(i); i['x']=clamp(float(i.get('x',rw/2)),w/2,max(w/2,rw-w/2)); i['z']=clamp(float(i.get('z',rl/2)),d/2,max(d/2,rl-d/2)); return i
def aabb(i):
    w,d,h=fp(i); x=float(i.get('x',0)); z=float(i.get('z',0)); return x-w/2,z-d/2,x+w/2,z+d/2
def overlap(a,b):
    ax1,az1,ax2,az2=aabb(a); bx1,bz1,bx2,bz2=aabb(b); dx=min(ax2,bx2)-max(ax1,bx1); dz=min(az2,bz2)-max(az1,bz1); return max(0,dx)*max(0,dz)
def snap_wall(i,room,wall=None):
    rw=float(room.get('widthM',4)); rl=float(room.get('lengthM',5)); w,d,h=fp(i); wall=wall or i.get('wallAnchor') or 'front'
    if wall=='left': i['x']=w/2; i['rotationY']=math.pi/2
    elif wall=='right': i['x']=rw-w/2; i['rotationY']=-math.pi/2
    elif wall=='back': i['z']=rl-d/2; i['rotationY']=math.pi
    else: i['z']=d/2; i['rotationY']=0; wall='front'
    i['wallAnchor']=wall; return snap_inside(i,room)
def find_support(item,items):
    iw,id,ih=fp(item); cand=[]
    for s in items:
        if s.get('productId')==item.get('productId'): continue
        if str(s.get('category','')).lower() not in SUPPORTER_CATEGORIES and not s.get('isSupporter',False): continue
        sw,sd,sh=fp(s)
        if sw>=iw and sd>=id: cand.append((((float(item.get('x',0))-float(s.get('x',0)))**2+(float(item.get('z',0))-float(s.get('z',0)))**2),s))
    return sorted(cand,key=lambda x:x[0])[0][1] if cand else None
def resolve_collisions(items,room):
    floor=[x for x in items if x.get('layer','floor')=='floor']
    for _ in range(60):
        moved=False
        for a_i in range(len(floor)):
            for b_i in range(a_i+1,len(floor)):
                a,b=floor[a_i],floor[b_i]; ov=overlap(a,b)
                if ov<=1e-3: continue
                dx=float(b.get('x',0))-float(a.get('x',0)); dz=float(b.get('z',0))-float(a.get('z',0)); n=(dx*dx+dz*dz)**.5 or 1; step=min(.2,ov**.5+.02); b['x']=float(b.get('x',0))+dx/n*step; b['z']=float(b.get('z',0))+dz/n*step; snap_inside(b,room); moved=True
        if not moved: break
    return items
def score(items,room):
    pen={'collision':0,'unsupported':0,'outside':0,'wall':0}
    for it in items:
        cat=str(it.get('category','')).lower()
        if cat in NEEDS_SUPPORT_CATEGORIES and it.get('layer')!='wall' and not it.get('supportParentId'): pen['unsupported']+=1
        if cat in WALL_CATEGORIES and not it.get('wallAnchor'): pen['wall']+=1
    for i in range(len(items)):
        for j in range(i+1,len(items)):
            if items[i].get('layer')=='floor' and items[j].get('layer')=='floor' and overlap(items[i],items[j])>1e-3: pen['collision']+=1
    return max(0,1-.1*pen['collision']-.12*pen['unsupported']-.05*pen['wall']),pen
def fast_solve_layout(room,raw_items):
    items=[dict(x) for x in raw_items]; rejected=[]
    for it in items:
        cat=str(it.get('category','unknown')).lower(); it.setdefault('footprint',{'widthM':float(it.get('widthM',.5)),'depthM':float(it.get('depthM',.5)),'heightM':float(it.get('heightM',.5))})
        if cat in WALL_CATEGORIES: it['layer']='wall'
        elif cat in FLOOR_UNDER_CATEGORIES: it['layer']='floor_under'
        elif cat in NEEDS_SUPPORT_CATEGORIES: it.setdefault('layer','top_surface')
        else: it.setdefault('layer','floor')
        snap_inside(it,room)
    for it in items:
        if it.get('layer')=='wall': snap_wall(it,room); it['y']=max(float(it.get('y',1.4)),1.2)
    for it in items:
        cat=str(it.get('category','')).lower()
        if cat in FLOOR_UNDER_CATEGORIES: it['x']=float(room.get('widthM',4))/2; it['z']=float(room.get('lengthM',5))/2; it['y']=.01; snap_inside(it,room)
        if cat in NEEDS_SUPPORT_CATEGORIES and it.get('layer')!='wall':
            sup=find_support(it,items)
            if sup is None: it['rejectReason']='needs_support_but_no_valid_supporter'; rejected.append(it)
            else: it['x']=sup.get('x',0); it['z']=sup.get('z',0); it['y']=float(sup.get('y',0))+fp(sup)[2]; it['supportParentId']=sup.get('productId')
    items=[x for x in items if not x.get('rejectReason')]; resolve_collisions(items,room)
    sc,pen=score(items,room); return items,rejected,{'fastScore':sc,'penalties':pen,'itemCount':len(items),'rejectedCount':len(rejected)}
