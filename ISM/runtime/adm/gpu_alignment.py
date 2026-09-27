"""Robust projective alignment with GPU normal equations and MPS Cholesky.

Input packing and spanning-tree initialization are small host setup operations.
The iterative objective, analytical derivatives, solves and updates run on device.
"""
from .io import write_array, write_json
import ctypes as C
import json
import time
import numpy as np
from .geometry import project


def solve(names, edges, out, width, gpu, iterations=60,workspace_budget_mb=512,symmetric=False):
    from .match_graph import require_connected
    require_connected(names,edges,out)
    started=time.perf_counter();n=len(names);anchor=n//2
    if n<2 or not edges or width<=0 or type(iterations) is not int or not 1<=iterations<=600:raise ValueError('Invalid GPU alignment input')
    if type(workspace_budget_mb) is not int or not 128<=workspace_budget_mb<=2048:raise ValueError('Invalid alignment workspace budget')
    transforms={anchor:np.eye(3)}
    while len(transforms)<n:
        candidates=[e for e in edges if (e['i'] in transforms)!=(e['j'] in transforms)]
        if not candidates:raise RuntimeError('Disconnected match graph; refusing to omit frames silently')
        e=max(candidates,key=lambda e:e['n']);i,j=e['i'],e['j']
        if i in transforms:transforms[j]=transforms[i]@np.linalg.inv(e['H'])
        else:transforms[i]=transforms[j]@e['H']
    S=np.diag([1/width,1/width,1.]);Si=np.linalg.inv(S)
    initial=np.array([S@transforms[i]@Si for i in range(n)])
    initial/=initial[:,2:3,2:3]
    x=gpu.upload(np.ascontiguousarray(initial.reshape(n,9)[:,:8],dtype=np.float32))
    matches=sum(len(e['a']) for e in edges);count=matches*(2 if symmetric else 1);d=n*8
    workspace=count*(32*4+16+4*3)+len(edges)*272*4+d*d*8+d*12
    if workspace>workspace_budget_mb*1024**2:
        raise ValueError(f'GPU alignment needs {workspace/1024**2:.1f} MiB, exceeding alignment_budget_mb={workspace_budget_mb}')
    points=np.empty((count,4),np.float32);owners=np.empty((count,2),np.uint32);edge_rows=[]
    links=np.full((n,n),-1,np.int32);incidents=[[] for _ in names];offset=0
    for k,e in enumerate(edges):
        i,j=e['i'],e['j'];m=len(e['a'])*(2 if symmetric else 1)
        if i==j or links[i,j]>=0:raise ValueError('Duplicate/self edges in alignment graph')
        pair=np.c_[e['a'],e['b']]
        points[offset:offset+m]=np.repeat(pair,2,axis=0) if symmetric else pair;owners[offset:offset+m]=[i,j]
        edge_rows.append([i,j,offset,m]);links[i,j]=links[j,i]=k
        incidents[i].append([k,0]);incidents[j].append([k,1]);offset+=m
    if not np.isfinite(points).all() or not np.isfinite(initial).all():raise ValueError('Nonfinite alignment input')
    offsets=np.r_[0,np.cumsum([len(v) for v in incidents])].astype(np.uint32)
    incidence=np.array([item for values in incidents for item in values],np.uint32)
    state=np.zeros(16+iterations*4,np.float32);state[0]=1e-3
    buffers=[x,gpu.upload(points),gpu.upload(owners),gpu.upload(np.array(edge_rows,np.uint32)),
             gpu.upload(links),gpu.upload(offsets),gpu.upload(incidence),gpu.upload(state)]
    prepared=time.perf_counter();before=gpu.stats()
    handles=(C.c_void_p*len(buffers))(*(b.handle for b in buffers))
    align=gpu.lib.adm_align_symmetric if symmetric else gpu.lib.adm_align
    gpu.check(align(gpu.handle,handles,n,len(edges),count,anchor,iterations,float(width)))
    finished=time.perf_counter();state=buffers[-1].numpy();values=x.numpy().astype(np.float64)
    hs={i:Si@np.r_[values[i],1.].reshape(3,3)@S for i in range(n)}
    if symmetric:
        errors=np.concatenate([np.linalg.norm(project(np.linalg.inv(hs[e[j]])@hs[e[i]],e[a])-e[b],axis=1)
                               for e in edges for i,j,a,b in [('i','j','a','b'),('j','i','b','a')]])
    else:errors=np.concatenate([np.linalg.norm(project(hs[e['i']],e['a'])-project(hs[e['j']],e['b']),axis=1) for e in edges])
    if not np.isfinite(errors).all():raise RuntimeError('GPU alignment produced invalid geometry')
    stats=gpu.stats()
    report=dict(anchor=names[anchor],images=n,edges=len(edges),observations=matches,residual_observations=count,
        median_registration_error_px=float(np.median(errors)),p95_registration_error_px=float(np.percentile(errors,95)),
        solver_success=bool(state[6]),solver_message=({1:'Device objective tolerance reached',2:'Device step tolerance reached',3:'Device predicted reduction tolerance reached'}.get(int(state[6]),'GPU iteration budget reached; inspect residuals before accepting')),
        solver='Metal analytical robust Gauss-Newton / adaptive damping / MPS Cholesky',
        objective='mean bidirectional soft_l1 transfer error in original registration-image pixels' if symmetric else 'soft_l1 per coordinate, 1 pixel scale, same correspondence set as CPU solver',
        relative_objective_tolerance=1e-5 if symmetric else 1e-6,
        convergence_requires_low_damping=bool(symmetric),
        iterations_executed=int(state[5]),iterations_encoded=int(state[9]),iterations_budget=iterations,
        convergence_check_interval=4,accepted_steps=int(state[2]),rejected_steps=int(state[3]),
        initial_objective=float(state[7]),final_objective=float(state[4]),factorization_status=int(state[8]),
        optimization_trace=state[16:16+int(state[5])*4].reshape(-1,4).tolist(),
        optimization_trace_columns=['old_objective','trial_objective','updated_damping','relative_step_norm'],
        host_setup_seconds=prepared-started,device_operation_wall_seconds=finished-prepared,
        gpu_seconds=stats['gpu_seconds']-before['gpu_seconds'],
        mps_operations=stats['mps_operations']-before['mps_operations'],
        command_submissions=stats['command_submissions']-before['command_submissions'],
        planned_scratch_bytes=workspace,
        workspace_budget_mb=workspace_budget_mb,
        cpu_work='Graph packing and initial spanning tree; final double-precision geographic conversion and diagnostics',
        jacobian='32 floats per point; only an 8N by 8N normal matrix, no dense per-pixel Jacobian')
    out.mkdir(parents=True,exist_ok=True)
    write_json(out/'registration_report.json',report);write_array(out/'transforms.npy',np.array([hs[i] for i in range(n)]))
    print(json.dumps(report,indent=2),flush=True)
    if not report['solver_success']:raise RuntimeError('GPU alignment did not converge; result is not accepted for compositing')
    return hs
