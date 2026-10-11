"""Enrich heading metadata without changing recorded positions, inputs or states."""
from phase1 import *
import h5py
def main():
    for run in ['pilot256_fast','full4096_fast']:
        out=ROOT/'datasets'/run;m=json.loads((out/'manifest.json').read_text())
        p,u,c,params=paths(m['seed'],m['per_condition'],160)
        with np.load(out/'trajectory_plan.npz') as old:
            assert np.array_equal(p,old['position']) and np.array_equal(u,old['model_input'])
        np.savez(out/'trajectory_plan.npz',position=p,model_input=u,condition=c,parameters=params,
                 sample_id=np.arange(len(p),dtype=np.int64))
        for r in m['files']:
            path=ROOT/r['path']
            with h5py.File(path,'r+') as f:
                ids=f['sample_id'][...];f['trajectory_parameters'][...]=params[ids]
                f.attrs['trajectory_parameter_columns']='initial_angle_rad,constant_speed_m_s,circle_radius_m,circle_center_x_m,circle_center_y_m,circle_period_steps'
                f.attrs['trajectory_parameter_semantics']='condition-specific: stationary none; constant angle/speed; circle angle/radius/center/period; random initial heading only; unused columns are zero'
            r.update(sha256=digest(path),bytes=path.stat().st_size)
        dump(out/'manifest.json',m)
        with (out/'samples.csv').open('w',newline='',encoding='utf-8') as f:
            w=csv.writer(f);w.writerow(['sample_id','condition','paired_initial_id','x0_m','y0_m','initial_angle_rad','constant_speed_m_s','circle_radius_m','circle_center_x_m','circle_center_y_m','circle_period_steps'])
            for i in range(len(p)):w.writerow([i,CONDITIONS[c[i]],i%m['per_condition'],*p[i,0],*params[i]])
        index(argparse.Namespace(run=run))
    print('Initial heading metadata complete; position/input arrays unchanged')
if __name__=='__main__':main()
