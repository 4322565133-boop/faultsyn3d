"""Audited Su pipeline in a common main-fault coordinate chart.

The three PSO points are the boundary anchors, not centres of independently
sized rectangles. Quadratic Bezier guide/profile curves are resampled on a
common physical X,Y chart. This preserves the planes and makes Y truncation
and final-surface intersection checks unambiguous. See docs/reproduction_v2.md
for numerical choices and limits of the discrete Algorithm-4 approximation.
"""
from __future__ import annotations

from dataclasses import asdict
import numpy as np
from scipy.ndimage import map_coordinates

from .geometry import (rotation_matrix, normal_from_points, dip_strike_from_normal,
                       bezier_control_point, bezier_curve)
from .optimize import axis_origins, axis_bounds, solve_pso, zeta_to_points
from . import stratigraphy as strat
from .displacement import hermite_bump, drag_operator


class RejectedSample(ValueError):
    pass


def _graph_curve(p0, p2, beta, coordinate, samples):
    """Eq. 4-9, alpha=1: invert a monotone physical-coordinate Bezier curve."""
    control = bezier_control_point(p0, p2, 1., beta)
    t = np.linspace(0, 1, max(256, len(samples)*2))
    points = bezier_curve(p0, control, p2, t)
    if np.any(np.diff(points[:, coordinate]) <= 1e-8):
        raise RejectedSample('nonmonotone_bezier')
    return np.interp(samples, points[:, coordinate], points[:, 2])


def surface_graph(anchors, xs, ys, beta_strike, beta_dip):
    """Two guide curves and physical-coordinate dip profiles (Fig. 4)."""
    p0, p1, p2 = anchors
    p3 = p1 + p2 - p0
    top = _graph_curve(p0, p1, beta_strike, 0, xs)
    bottom = _graph_curve(p2, p3, beta_strike, 0, xs)
    h = np.empty((len(xs), len(ys)))
    for i, x in enumerate(xs):
        h[i] = _graph_curve(np.array([x, ys[0], top[i]]),
                            np.array([x, ys[-1], bottom[i]]), beta_dip, 1, ys)
    return h


def _triangles_disjoint(a, b, clearance=0.):
    """Exact separating-axis test for two 2-D control triangles, including lines."""
    edges=np.concatenate([np.roll(a,-1,axis=0)-a,np.roll(b,-1,axis=0)-b])
    axes=np.vstack([np.column_stack([-edges[:,1],edges[:,0]]),edges])
    norms=np.linalg.norm(axes,axis=1)
    axes=axes[norms>1e-10]/norms[norms>1e-10,None]
    if not len(axes):return bool(np.linalg.norm(a[0]-b[0])>clearance)
    pa,pb=a@axes.T,b@axes.T
    return bool(np.any((pa.max(0)+clearance<pb.min(0)-1e-9)|
                       (pb.max(0)+clearance<pa.min(0)-1e-9)))


def check_control_domains(tree, anchors, betas, valid, xs, ys):
    """Eq. 53 / Algorithm 4 on the two reference control triangles.

    We sample finite singleton domains. Reference curves are restricted to
    their surviving Y intervals by exact quadratic subdivision; otherwise
    a grandchild would be rejected for intersections on its discarded half.
    Interval endpoints have a one-grid-cell guard. The 3-D interior is checked
    separately. This finite-domain numerical interpretation is documented.
    """
    tested=0
    for k in (1,2):
        forest=tree.direction_forest(k)
        triangles=[]
        axes=[0,2] if k==1 else [1,2]
        for i in range(len(tree)):
            p0,pk=anchors[i,0],anchors[i,k]
            ctrl=bezier_control_point(p0,pk,1.,betas[i,k-1])
            full=np.array([p0,ctrl,pk])
            mask=valid[i,:,0] if k==1 else valid[i,0,:]
            samples=xs if k==1 else ys
            starts=np.flatnonzero(mask&~np.r_[False,mask[:-1]])
            ends=np.flatnonzero(mask&~np.r_[mask[1:],False])
            t=np.linspace(0,1,1025)
            curve=bezier_curve(*full,t)
            pieces=[]
            for start,end in zip(starts,ends):
                aa=samples[max(0,start-1)];bb=samples[min(len(samples)-1,end+1)]
                ta,tb=np.interp([aa,bb],curve[:,k-1],t)
                q0,q2=bezier_curve(*full,np.array([ta,tb]))
                deriv=2*((1-ta)*(ctrl-p0)+ta*(pk-ctrl))
                q1=q0+.5*(tb-ta)*deriv
                pieces.append(np.array([q0,q1,q2])[:,axes])
            for j,others in enumerate(triangles):
                if forest.nodes[i].parent_id==j:continue
                for tri in pieces:
                    for other in others:
                        tested+=1
                        if not _triangles_disjoint(tri,other):
                            raise RejectedSample('empty_sampled_control_domain')
            triangles.append(pieces)
    return tested


def _interp(a, u, v, xs, ys, order=1):
    ij = np.array([(u-xs[0])/(xs[-1]-xs[0])*(len(xs)-1),
                   (v-ys[0])/(ys[-1]-ys[0])*(len(ys)-1)])
    return map_coordinates(a, ij, order=order, mode='nearest', prefilter=False)


def validate_surfaces(tree, heights, xs, ys, centre, R, grid, z, supports=None):
    """Check final graphs, including Y clipping and domain-limited crossings.

    X must cross, Y must terminate on an extant parent, and non-parent
    surfaces may not develop an interior crossing. Near a common Y junction
    contact is allowed. All checks run before volume rasterisation.
    """
    U,V = np.meshgrid(xs, ys, indexing='ij')
    n = len(tree)
    base = np.ones((n, len(xs), len(ys)), dtype=bool) if supports is None else supports.copy()
    valid = base.copy()
    for i in range(1,n):
        node = tree.nodes[i]
        if 'Y' in node.in_edge:
            p = node.parent_id
            sign = np.sign(z[i,0]-z[p,0])
            if sign == 0: raise RejectedSample('zero_Y_reference_side')
            # Retain the wall opposite P0. This is the side consistent with
            # Algorithm 1's rho ordering: the Y segment starts at its parent
            # junction, beyond the reference P0 side. Keeping P0 instead
            # retains the forbidden grandparent crossing in deep trees.
            valid[i] &= (heights[i]-heights[p])*sign <= 1e-9
            valid[i] &= valid[p]
    inside = []
    for i,h in enumerate(heights):
        pts = np.stack([U,V,h],axis=-1) @ R + centre
        mask = np.all((pts >= 0)&(pts <= np.asarray(grid)-1),axis=-1)
        inside.append(mask)
        if np.count_nonzero(mask&valid[i]) < max(8,int(160*(len(xs)/193)**2)):
            raise RejectedSample('missing_or_tiny_surface')
    edges = []
    tolerance = 1.5  # grid sampling tolerance in chart-normal voxel units
    for i in range(1,n):
        node = tree.nodes[i]; p=node.parent_id
        d = heights[i]-heights[p]
        # The intersection itself must be inside both finite, clipped domains.
        overlap = inside[i]&inside[p]&valid[p]&base[i]
        if not overlap.any(): raise RejectedSample('parent_outside_domain')
        if 'Y' in node.in_edge or 'X' in node.in_edge:
            # Cell sign crossings are more reliable than a nearest sampled gap.
            cross_u = (d[:-1]*d[1:] <= 0)&(abs(d[:-1]-d[1:])>1e-9)&overlap[:-1]&overlap[1:]
            cross_v = (d[:,:-1]*d[:,1:] <= 0)&(abs(d[:,:-1]-d[:,1:])>1e-9)&overlap[:,:-1]&overlap[:,1:]
            crossings=int(cross_u.sum()+cross_v.sum())
            if crossings < max(2,int(8*len(xs)/193)): raise RejectedSample('missing_required_intersection')
            if 'Y' in node.in_edge:
                contact = overlap&valid[i]&(abs(d)<tolerance)
                if contact.sum()<max(1,int(4*len(xs)/193)):raise RejectedSample('missing_Y_contact')
            edges.append(dict(parent=p,child=i,edge=list(node.in_edge),crossing_cells=crossings))
        else:
            if d[overlap].min() <= 0 <= d[overlap].max():
                raise RejectedSample('P_crossing')
            gu,gv=np.gradient(heights[i],xs,ys)
            gap=abs(d)/np.sqrt(1+gu*gu+gv*gv)
            if np.min(gap[overlap])<2.25:raise RejectedSample('P_too_close')
            edges.append(dict(parent=p,child=i,edge=list(node.in_edge),min_chart_gap=float(np.min(abs(d[overlap])))))
    for i in range(n):
        for j in range(i+1,n):
            if tree.nodes[j].parent_id == i:continue
            overlap=inside[i]&inside[j]&valid[i]&valid[j]
            if not overlap.any():continue
            d=(heights[i]-heights[j])[overlap]
            if d.min() < -tolerance and d.max() > tolerance:
                raise RejectedSample(f'unintended_surface_crossing_{i}_{j}')
    return valid, inside, edges


def generate(tree, main, cfg, rng, category):
    """Generate one accepted 128^3 volume, or report an explicit rejection."""
    grid=tuple(cfg['grid']); nx,ny,nz=grid
    centre=np.array(main.centre); R=rotation_matrix(main.strike_deg,main.dip_deg)
    hl,hw=main.half_len,main.half_wid
    corners={
        'G0_top':np.array([-hl,-hw,0.])@R+centre,
        'G2_top':np.array([hl,-hw,0.])@R+centre,
        'G0_bottom':np.array([-hl,hw,0.])@R+centre,
        'G2_bottom':np.array([hl,hw,0.])@R+centre}
    origins=axis_origins(centre,R,corners)
    lo,hi=axis_bounds(grid,centre,R,origins,cfg['d_z'])
    zz,info=solve_pso(tree,lo,hi,rng,n_particles=cfg['pso_particles'],max_iter=cfg['pso_iters'])
    if info['penalty'] != 0:raise RejectedSample('pso_nonzero')
    z=np.vstack([np.zeros(3),zz]); n=len(tree)
    anchors=np.repeat(origins[None],n,axis=0)
    anchors[:,:,2]+=z*cfg['d_z']
    frames=[]; faults=[]
    for i in range(n):
        points=anchors[i]@R+centre
        normal=normal_from_points(*points)
        dip,strike=dip_strike_from_normal(normal)
        frame=rotation_matrix(strike,dip)
        residual=float(np.max(abs((points-points.mean(0))@frame[2])))
        if residual>1e-7:raise AssertionError('plane roundtrip failed')
        frames.append(frame)
        faults.append(dict(id=i,dip=dip,strike=strike,anchors=points.tolist(),
                           plane_residual=residual,in_edge=tree.nodes[i].in_edge,
                           parent=tree.nodes[i].parent_id))
    ns=cfg['surface_samples']
    xs=np.linspace(-hl,hl,ns); ys=np.linspace(-hw,hw,ns)
    U,V=np.meshgrid(xs,ys,indexing='ij')
    supports=np.ones((n,ns,ns),bool);support_info=[]
    for i in range(n):
        node=tree.nodes[i]
        # Finite branch extents are unspecified by the paper. Define patches
        # in the shared chart, preserving their three-point planes exactly.
        # X branches need broad support; Y splays use finite independent tips.
        if i and 'Y' in node.in_edge:
            cu=float(rng.uniform(*cfg['y_support_center_u'])*hl);cv=float(rng.uniform(*cfg['y_support_center_v'])*hw)
            hu=float(rng.uniform(*cfg['y_support_half_u'])*hl);hv=float(rng.uniform(*cfg['y_support_half_v'])*hw)
            supports[i]=(abs(U-cu)<=hu)&(abs(V-cv)<=hv)
            support_info.append(dict(id=i,centre_uv=[cu,cv],half_extent_uv=[hu,hv]))
        else:support_info.append(dict(id=i,centre_uv=[0.,0.],half_extent_uv=[hl,hw]))
    # Node attributes are correlated within a network, a declared sampling
    # choice. They are still quadratic curves; no surface is moved after PSO.
    bs=rng.uniform(*cfg['beta_strike'])
    bd=rng.uniform(*(cfg['beta_listric'] if category=='listric_assemblage' else cfg['beta_dip']))
    if category!='listric_assemblage':bd*=rng.choice([-1.,1.])
    bs*=rng.choice([-1.,1.])
    jitter=rng.uniform(*cfg['curvature_jitter'],size=(n,2));jitter[0]=1
    failure='no_curvature_domain'
    for scale in cfg['curvature_scales']:
        try:
            # Cheap rejection before allocating the dense chart. Every
            # survivor is checked again at full resolution below.
            coarse_x=np.linspace(-hl,hl,33);coarse_y=np.linspace(-hw,hw,33)
            coarse_support=supports[:,::max(1,(ns-1)//32),::max(1,(ns-1)//32)]
            if coarse_support.shape[1:] == (33,33):
                coarse_h=np.array([surface_graph(anchors[i],coarse_x,coarse_y,bs*scale*jitter[i,0],bd*scale*jitter[i,1]) for i in range(n)])
                validate_surfaces(tree,coarse_h,coarse_x,coarse_y,centre,R,grid,z,coarse_support)
            heights=np.array([surface_graph(anchors[i],xs,ys,bs*scale*jitter[i,0],bd*scale*jitter[i,1]) for i in range(n)])
            valid,inside,edges=validate_surfaces(tree,heights,xs,ys,centre,R,grid,z,supports)
            domain_tests=check_control_domains(tree,anchors,jitter*np.array([bs,bd])*scale,valid,xs,ys)
            break
        except RejectedSample as e:failure=str(e)
    else:raise RejectedSample(failure)

    # Dense chart grids are retained in export; invalid parts are encoded by
    # the same mask used for displacement and labels.
    U,V=np.meshgrid(xs,ys,indexing='ij')
    surfaces={}; normals=[]; slips=[]; areas=[]
    gradients=[]
    for i,h in enumerate(heights):
        hu,hv=np.gradient(h,xs,ys)
        gradients.append((hu,hv))
        jac=np.sqrt(1+hu*hu+hv*hv)
        area=float(np.sum(jac*valid[i]*inside[i])*(xs[1]-xs[0])*(ys[1]-ys[0]))
        areas.append(area)
        nn=np.stack([-hu,-hv,np.ones_like(h)],axis=-1)/jac[...,None]
        normals.append(nn@R)
        pts=np.stack([U,V,h],axis=-1)@R+centre
        surfaces[i]=np.where(valid[i,...,None],pts,np.nan).astype(np.float32)
        # Physical X/Y coordinates, not an uncalibrated Bezier parameter t.
        # Normalise each remaining row/column on the finite, truncated patch.
        loc=(pts-np.mean(faults[i]['anchors'],axis=0))@frames[i].T
        def normalised(a,axis):
            low=np.min(np.where(valid[i],a,np.inf),axis=axis,keepdims=True)
            high=np.max(np.where(valid[i],a,-np.inf),axis=axis,keepdims=True)
            ok=np.isfinite(low)&np.isfinite(high)&(high>low+1e-6)
            span=np.where(ok,high-low,1.)
            return np.where(valid[i]&ok,2*(a-np.where(ok,low,0.))/span-1,1.)
        sx=normalised(loc[...,0],0);sy=normalised(loc[...,1],1)
        slip=hermite_bump(sx)*hermite_bump(sy)
        # Throughgoing main fault uses Eq. 14; all other finite surfaces Eq. 16.
        if i==0 and main.throughgoing:
            rr=np.clip(np.sqrt(sx*sx+sy*sy),0,1)
            slip=2*(1-rr)*np.sqrt(np.maximum((1+rr)**2/4-rr*rr,0))
        slips.append(np.where(valid[i],slip,0.))
    if min(areas)<cfg['min_area']:raise RejectedSample('insufficient_effective_area')
    dmax=main.d_max*np.sqrt(np.array(areas)/areas[0])
    for i,f in enumerate(faults):
        f.update(area=areas[i],d_max=float(dmax[i]),beta_strike=bs*scale*jitter[i,0],
                 beta_dip=bd*scale*jitter[i,1],alpha=1.)

    gz,gy,gx=np.meshgrid(np.arange(nz,dtype=np.float32),np.arange(ny,dtype=np.float32),
                          np.arange(nx,dtype=np.float32),indexing='ij')
    shape=gx.shape
    q=np.stack([gx.ravel()-centre[0],gy.ravel()-centre[1],gz.ravel()-centre[2]],axis=0)
    q=R@q;u,v,w=q
    chart=(u>=xs[0])&(u<=xs[-1])&(v>=ys[0])&(v<=ys[-1])
    disp=np.zeros((3,len(w)),np.float32)
    label=np.zeros(len(w),np.uint8);instances=np.zeros(len(w),np.uint8)
    confidence=np.zeros(len(w),np.float32)
    freq=float(rng.uniform(*cfg['freq_hz'])); threshold=.25/(freq*cfg['dt'])
    for i,h in enumerate(heights):
        hu,hv=gradients[i]
        hh=_interp(h,u,v,xs,ys);gu=_interp(hu,u,v,xs,ys);gv=_interp(hv,u,v,xs,ys)
        # Normal projection of an implicit height field; refined below for
        # near-surface voxels to remove orientation-dependent point sampling.
        jac=np.sqrt(1+gu*gu+gv*gv)
        sd=(w-hh)/jac
        active=chart&(_interp(valid[i].astype(float),u,v,xs,ys,order=0)>.5)
        reach=active&(abs(sd)<=cfg['drag_distance'])
        ix=np.flatnonzero(reach)
        if not len(ix):raise RejectedSample('empty_raster')
        ksi=_interp(slips[i],u[ix],v[ix],xs,ys)*dmax[i]
        direction=np.array([np.sin(np.deg2rad(main.phi_dis_deg)),np.cos(np.deg2rad(main.phi_dis_deg)),0.])@frames[i]
        reference_sign=np.sign(np.dot(frames[i][2],R[2])) or 1.
        df=drag_operator(sd[ix]*reference_sign/cfg['drag_distance'],main.drag_mode)
        disp[:,ix]+=(direction[:,None]*ksi[None,:]*df[None,:]).astype(np.float32)
        # Refine the narrow label band by minimising point-to-graph distance
        # in the orthonormal root chart (Gauss-Newton, four iterations).
        near=np.flatnonzero(abs(sd[ix])<=cfg['label_half_thickness']+1.)
        vi0=ix[near]; pu=u[vi0].copy();pv=v[vi0].copy()
        for _ in range(4):
            ph=_interp(h,pu,pv,xs,ys);a1=_interp(hu,pu,pv,xs,ys);a2=_interp(hv,pu,pv,xs,ys)
            du=pu-u[vi0]+(ph-w[vi0])*a1;dv=pv-v[vi0]+(ph-w[vi0])*a2
            factor=(du*a1+dv*a2)/(1+a1*a1+a2*a2)
            pu=np.clip(pu-(du-a1*factor),xs[0],xs[-1])
            pv=np.clip(pv-(dv-a2*factor),ys[0],ys[-1])
        ph=_interp(h,pu,pv,xs,ys)
        distance=np.sqrt((pu-u[vi0])**2+(pv-v[vi0])**2+(ph-w[vi0])**2)
        on=np.zeros(len(ix),bool)
        on[near]=(distance<=cfg['label_half_thickness'])&(_interp(valid[i].astype(float),pu,pv,xs,ys,order=0)>.5)
        vi=ix[on]
        label[vi]=1;instances[vi]=i+1
        # A diagnostic, never a hard deletion rule. Fig. 7 has a +/-1 jump.
        jump=2*abs(direction[2])*ksi[on]
        confidence[vi]=np.maximum(confidence[vi],np.clip(jump/threshold,0,1))
        faults[i]['label_voxels']=int(len(vi))
        faults[i]['visible_fraction']=float(np.mean(jump>=threshold)) if len(vi) else 0.
        if len(vi)<cfg['min_instance_voxels']:raise RejectedSample('empty_label_instance')
        if faults[i]['visible_fraction']<cfg['min_visible_fraction']:
            raise RejectedSample('whole_branch_weak_response')

    # Eq. 56 additive coordinate mapping on a 1-D reflectivity sequence.
    # Horizontal displacements are stored diagnostically; no unsupported claim
    # of full elastic modelling or horizontal reflectivity variation is made.
    a,b=rng.uniform(-cfg['tilt'],cfg['tilt'],2)
    folds=strat.random_folds(rng,int(rng.integers(*cfg['n_folds'])),nx,ny,
                             tuple(cfg['fold_amp']),tuple(cfg['fold_sigma']))
    zdef=gz+strat.linear_shift(gx,gy,a,b,centre[0],centre[1])+strat.fold_shift(gx,gy,gz,folds,nz)+disp[2].reshape(shape)
    margin=int(np.ceil(max(abs(float(zdef.min())),abs(float(zdef.max())-nz))))+32
    ref=strat.reflectivity_1d(nz+2*margin,rng,n_layers=(100,220))
    reflectivity=strat.sample_reflectivity(ref,zdef+margin)
    clean=strat.convolve_z(reflectivity,strat.ricker(freq,cfg['dt']))
    snr=float(rng.uniform(*cfg['snr_db']))
    seismic=strat.normalize(strat.add_noise(clean,snr,rng))
    meta=dict(category=category,grid=list(grid),main=asdict(main),faults=faults,
              zeta=zz.tolist(),pso=info,n_faults=n,tree_depth=tree.depth(),
              tree=[dict(id=node.id,parent=node.parent_id,in_edge=node.in_edge) for node in tree.nodes.values()],
              fault_fraction=float(label.mean()),fault_fraction_full=float(label.mean()),
              observable_labels=False,curvature_scale=scale,topology_checks=edges,
              geometry_qc_pass=True,plane_residual_max=max(f['plane_residual'] for f in faults),
              label_visible_fraction=float(np.mean(confidence[label>0]>=1)),
              freq_hz=freq,snr_db=snr,folds=folds,tilt=[a,b],background_margin=margin,
              generator='su_reproduction_v2',label_distance='point_to_graph_gauss_newton_4',
              config=cfg,control_hull_tests=domain_tests,
              finite_support=support_info,
              curvature_domain_method='singleton_domains_exact_2d_SAT_then_final_surface_validation')
    return dict(seismic=seismic,label=label.reshape(shape),label_full=label.reshape(shape),
                instances=instances.reshape(shape),confidence=(255*confidence.reshape(shape)).astype(np.uint8),
                meta=meta,surfaces=surfaces)
