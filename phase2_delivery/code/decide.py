from common import *
def holm(values):
    order=np.argsort(values);out=np.zeros(len(values));last=0.
    for j,i in enumerate(order):last=max(last,min(1.,(len(values)-j)*values[i]));out[i]=last
    return out.tolist()
def main():
    source='official_ganguli_pretrained';base=P2/'analysis'/source
    results={}
    for split in ['validation','confirmation']:
        slots=[]
        for s in ['existing_full4096_fast',source]:
            p=P2/'analysis'/s/f'topology/{split}_primary_test.json'
            slots.append((f'v1:{s}:foraging',json.loads(p.read_text())['p_unadjusted'] if p.exists() else 1.))
        for v in ['v2','v3']:
            for s in ['existing_full4096_fast',source]:
                for c in ['stationary','random_foraging']:
                    p=P2/'analysis'/s/f'{v}/topology/{c}_{split}_test.json'
                    slots.append((f'{v}:{s}:{c}',json.loads(p.read_text())['p_unadjusted'] if p.exists() else 1.))
        adjusted=holm([x[1] for x in slots]);results[split]={name:dict(raw_p=p,Holm_p=adj) for (name,p),adj in zip(slots,adjusted)}
    accepted={}
    for c in ['stationary','random_foraging']:
        both=[]
        for split in ['validation','confirmation']:
            p=base/f'v3/topology/{c}_{split}_test.json'
            if not p.exists():both.append(False);continue
            r=json.loads(p.read_text());key=f'v3:{source}:{c}'
            both.append(r['observed']['signature_pass'] and r['field_and_resolution_pass'] and results[split][key]['Holm_p']<=.05)
        accepted[c]=all(both)
    report=dict(acceptance_by_condition=accepted,Holm_family_size=10,all_method_attempts_accounted=True,tests=results,
       source=source,neuron_count=483,all_neuron_population_torus_claim=False,Schoyen_three_environment_reproduction=False,
       strict_homeomorphism_proven=False,interpretation='finite-sample torus-consistent raw subpopulation topology in the accepted condition only')
    save_json(P2/'RESULTS.json',report);print(json.dumps(report),flush=True)
if __name__=='__main__':main()
