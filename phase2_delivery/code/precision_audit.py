from topology import *
from scipy.stats import beta
def main():
    source='official_ganguli_pretrained';out=P2/'analysis'/source
    cells=np.load(out/'paper_cluster_candidate_v2.npy');train=np.load(out/'discovery_states_step20.npy',mmap_mode='r')
    mean,rms=transform_fit(train,cells);dest=out/'v3/precision_audit';dest.mkdir(exist_ok=True)
    for split,seed in [('validation',73101),('confirmation',73102)]:
        original=out/f'v3/topology/stationary_{split}_test.json'
        if not original.exists():raise RuntimeError('Complete the original 199-draw test first')
        primary=json.loads(original.read_text());obs=primary['observed']
        if not obs['signature_pass'] or primary['p_unadjusted']>.005:
            save_json(dest/f'{split}.json',dict(status='not_used_to_rescue_failed_primary'));continue
        raw=np.load(out/f'v3/{split}_stationary_states.npy',mmap_mode='r');x=(raw[:,cells]-mean)/rms
        rng=np.random.default_rng(seed);scores=[];methods=[]
        for b in range(999):
            sh=np.empty_like(x)
            for j in range(x.shape[1]):sh[:,j]=x[rng.permutation(len(x)),j]
            r=null_ph(sh,n=768);scores.append(r['score'])
            methods.append('full_H2' if r['H2_computed'] else ('H1_only' if r.get('H1_computed',True) else 'edge_bound'))
            if b%50==0:
                save_json(dest/f'{split}_progress.json',dict(completed=b+1,planned=999,scores=scores,methods=methods))
                print('precision',split,b+1,'/ 999',flush=True)
        exceed=int(np.sum(np.array(scores)>=obs['score']));p=(1+exceed)/1000
        upper=float(beta.ppf(.95,exceed+1,999-exceed)) if exceed<999 else 1.
        save_json(dest/f'{split}.json',dict(B=999,p_plus_one=p,exceedances=exceed,
             one_sided_95_percent_binomial_tail_upper=upper,conservative_times10_p=min(1,10*p),
             conservative_times10_tail_upper=min(1,10*upper),scores=scores,methods=methods,
             primary_199_required=True,can_rescue_primary=False,seed=seed))
        print('precision completed',split,p,upper,flush=True)
if __name__=='__main__':setup();main()
