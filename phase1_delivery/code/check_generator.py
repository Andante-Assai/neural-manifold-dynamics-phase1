"""Check vectorized square wall rule against original scalar Agent/Rectangle."""
from phase1 import *
sys.path.insert(0,str(ROOT/'rat_original'))
from ratsimulator import Agent
from ratsimulator.Environment import Rectangle as AuthorRectangle

def main():
    rng=np.random.default_rng(876)
    points=np.concatenate([rng.uniform(-.02,BOX+.02,(500,2)),[[.001,.001],[2.199,.001],[.001,2.199],[2.199,2.199],[1.1,1.1]]])
    headings=rng.uniform(0,2*np.pi,len(points));speed=rng.rayleigh(.13*2*np.pi,len(points))*DT;turn=rng.normal(0,11.52,len(points))*DT
    env=AuthorRectangle(boxsize=(BOX,BOX),soft_boundary=.03);agent=Agent(env,boundary_mode='sorschers')
    clip=np.clip(points,1e-8,BOX-1e-8)
    ed=(clip<=.03).astype(float)-(clip>=BOX-.03).astype(float)
    orig_ed=np.array([env.wall_rejection(x) for x in clip])
    assert np.allclose(ed,orig_ed,atol=1e-7) # scalar projection has ~2e-8 cancellation near clipped walls
    angle=headings+turn;dirs=np.stack([np.cos(angle),np.sin(angle)],-1)
    mask=(ed*dirs).sum(-1)<0
    sp=(ed*np.stack([np.cos(angle+np.pi/2),np.sin(angle+np.pi/2)],-1)).sum(-1)
    sn=(ed*np.stack([np.cos(angle-np.pi/2),np.sin(angle-np.pi/2)],-1)).sum(-1)
    newturn=turn+mask*np.sign(sp-sn)*np.pi/2;newspeed=np.where(mask,speed/4,speed)
    original=np.array([agent.sorschers_avoid_walls(x,hd,s,t) for x,hd,s,t in zip(points,headings,speed,turn)])
    assert np.allclose(np.column_stack([newspeed,newturn]),original,atol=1e-12)
    pos,u,c,params=paths(20261011,512,160)
    assert np.array_equal(pos[c==0],np.repeat(pos[c==0,:1],161,axis=1))
    assert np.max(np.linalg.norm(pos[c==2,20]-pos[c==2,0],axis=-1))<1e-6
    assert np.max(np.abs(np.diff(u[c==1,1:],axis=1)))<5e-7
    assert np.allclose(np.diff(pos,axis=1),u[:,1:])
    dump(ROOT/'audit'/'generator_verification.json',dict(status='passed',author_boundary_cases=len(points),
       rejection_vectors_match=True,scalar_speed_turn_match=True,stationary_exact=True,
       constant_displacement=True,closed_circle_period20=True,input_position_alignment=True,
       caveat='Tests establish equation and controlled-path semantics; stochastic RNG ordering is not the author serial generator ordering.'))
    print('Generator checks passed')
if __name__=='__main__':main()
