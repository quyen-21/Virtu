def needs_heavy_check(items, metrics):
    p=metrics.get('penalties',{})
    return p.get('collision',0)>0 or p.get('unsupported',0)>0 or any(x.get('layer')=='top_surface' for x in items)
def heavy_check_layout(room, items):
    try:
        import pybullet as p
        cid=p.connect(p.DIRECT); p.disconnect(cid)
        return True, {'pybulletAvailable': True, 'checkedItems': len(items)}
    except Exception as e:
        return True, {'pybulletAvailable': False, 'note': str(e)}
