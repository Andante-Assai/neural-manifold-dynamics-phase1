from topology import *
@torch.no_grad()
def main():
    source='official_ganguli_pretrained';out=P2/'analysis'/source;dest=out/'geometry_support';dest.mkdir(exist_ok=True)
    cells=np.load(out/'paper_cluster_candidate_v2.npy');model,ck=get_model(source,setup(73010))
    rng=np.random.default_rng(73010);positions=rng.uniform(.05,2.15,(256,2)).astype(np.float32)
    jac=[]
    for eps in [.001,.0005]:
        points=np.concatenate([positions+[eps,0],positions-[eps,0],positions+[0,eps],positions-[0,eps]]).astype(np.float32)
        pt=torch.as_tensor(points,device='cuda');u=torch.zeros((len(points),21,2),device='cuda');p0=model.pc(pt)
        h=model(u,p0)[:,-1,cells].cpu().numpy()
        dx=(h[:256]-h[256:512])/(2*eps);dy=(h[512:768]-h[768:])/(2*eps)
        J=np.stack([dx,dy],-1);s=np.linalg.svd(J,compute_uv=False);jac.append(s)
    np.savez(dest/'position_Jacobian_singular_values.npz',position=positions,eps=[.001,.0005],singular_values=np.array(jac))
    annotations=json.loads((out/'discovery_cluster_annotations.json').read_text());sel=json.loads((out/'paper_cluster_selection_v2.json').read_text())['chosen_triplet']['cluster_labels']
    chosen=[next(r for r in annotations if r['cluster']==i) for i in sel]
    K=np.array([[r['frequency_median']*np.cos(np.deg2rad(r['angle_deg'])),r['frequency_median']*np.sin(np.deg2rad(r['angle_deg']))] for r in chosen[:2]])/2.2
    A=np.linalg.inv(K);results=[]
    for axis in range(2):
        predicted=A[:,axis];control=np.array([[np.cos(np.pi/4),-np.sin(np.pi/4)],[np.sin(np.pi/4),np.cos(np.pi/4)]])@predicted
        for label,shift in [('discovery_predicted_lattice',predicted),('same_length_rotated_control',control)]:
            lo=np.maximum(.05,.05-shift);hi=np.minimum(2.15,2.15-shift)
            p=rng.uniform(lo,hi,(512,2)).astype(np.float32);points=np.r_[p,p+shift].astype(np.float32)
            pt=torch.as_tensor(points,device='cuda');u=torch.zeros((len(points),21,2),device='cuda')
            hh=model(u,model.pc(pt))[:,-1].cpu().numpy()
            delta=np.linalg.norm(hh[:512,cells]-hh[512:,cells],axis=1)
            full=np.linalg.norm(hh[:512]-hh[512:],axis=1)
            results.append(dict(axis=axis,type=label,physical_shift_m=shift.tolist(),
                         candidate_difference_median=float(np.median(delta)),candidate_difference_mean=float(delta.mean()),
                         full_population_difference_median=float(np.median(full))))
    save_json(dest/'geometry_support.json',dict(Jacobian_locally_rank2_fraction=float(np.mean(jac[-1][:,1]/jac[-1][:,0]>1e-3)),
       Jacobian_second_to_first_singular_ratio_median=float(np.median(jac[-1][:,1]/jac[-1][:,0])),
       Jacobian_note='Stationary states are a deterministic map of two physical initial-position variables. Local rank supports 2D organization, not global homeomorphism.',
       predicted_lattice_matrix_m=A.tolist(),periodicity_tests=results,
       scope='supporting tests, no neuron selection or PH threshold fitted from these results'))
    # IID rather than farthest-point samples for intrinsic-dimension estimates.
    dims=[]
    for split in ['validation','confirmation']:
        for cond in ['stationary','random_foraging']:
            x=np.load(out/f'v3/{split}_{cond}_states.npy',mmap_mode='r')[:,cells]
            inds=np.random.default_rng(73017).choice(len(x),4096,replace=False);d=distances(x[inds]);np.fill_diagonal(d,np.inf)
            near=np.partition(d,39,axis=1)[:,:40];near.sort(1)
            good=near[:,0]>1e-8;logs=np.log(near[good,1]/near[good,0]);logs=np.sort(logs)
            logs=logs[int(.1*len(logs)):int(.9*len(logs))]
            twonn=float(1/np.mean(logs));ml={}
            for k in [10,20,40]:ml[str(k)]=float(1/np.mean(np.log(near[good,k-1:k]/near[good,:k-1])))
            dims.append(dict(split=split,condition=cond,n_IID_sampled=4096,TwoNN_trimmed=twonn,Levina_Bickel=ml,
                   note='finite-neighborhood estimates can reflect thickness, overlapping sheets and curvature; FPS-based TwoNN in earlier diagnostics is sampling-biased'))
    save_json(dest/'IID_intrinsic_dimensions.json',dims)
    print('Local rank, predicted-lattice periodicity and IID dimension diagnostics complete',flush=True)
if __name__=='__main__':main()
