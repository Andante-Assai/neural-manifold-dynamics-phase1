"""Offline report of inspected public artifacts; no assertion about private availability."""
from phase1 import *
import xml.etree.ElementTree as ET
def main():
    a=ROOT/'audit'
    trees={name:json.loads((a/f'{name}.json').read_text()) for name in ['github-tree','github-v1-tree','rat-tree']}
    releases=json.loads((a/'github-releases.json').read_text())
    z=json.loads((a/'zenodo-code.json').read_text())
    current={x['path']:x for x in trees['github-tree']['tree'] if x['type']=='blob'}
    v1={x['path']:x for x in trees['github-v1-tree']['tree'] if x['type']=='blob'}
    extensions=('.pt','.pth','.pkl','.npy','.npz','.h5','.hdf5','.ckpt','.safetensors')
    found={k:[x['path'] for x in tree['tree'] if x['path'].lower().endswith(extensions)] for k,tree in trees.items()}
    model_files=['src/Models/SorscherRNN.py','src/PlaceCells.py','src/Logger.py','src/datahandling.py']
    comparison={p:current[p]['sha']==v1[p]['sha'] for p in model_files}
    report=dict(inspected_on='2026-10-11 Asia/Shanghai',paper='https://pmc.ncbi.nlm.nih.gov/articles/PMC10589895/',
        author_repo='https://github.com/bioAI-Oslo/emergent-grid-cells',
        github_main_commit=trees['github-tree']['sha'],publication_v1_commit=trees['github-v1-tree']['sha'],
        rat_simulator_commit=trees['rat-tree']['sha'],model_files_identical_main_vs_v1=comparison,
        matching_checkpoint_or_data_files=found,release_assets=[r['assets'] for r in releases],
        zenodo_record='https://zenodo.org/records/8318009',zenodo_files=z['files'],
        availability_conclusion='No public training weights or trajectory arrays in inspected repository main/v1 trees or release assets. Zenodo lists only a code zip. This does not establish that weights are unavailable privately or in uninspected locations.',
        missing_default_input='Experiment._init_default_experiment requires example_pc_centers.npy, absent from inspected repo. Local run samples a seeded uniform place layout using the author PlaceCells class.',
        archive_download_note='GitHub commit zip successfully downloaded and extracted. Zenodo code-zip transfer ended prematurely; metadata retained. No need to substitute this partial archive for source.',
        paper_settings=dict(Ng=4096,Np=512,T=20,dt=.02,box_m=[2.2,2.2],batch=200,lr=.0001,recurrent_L2=.0001,
                            paper_main_environments=3,minibatches_per_environment=100000),
        implementation_notes=['Author SorscherRNN and PlaceCells imported without editing them.',
            'Use author loss_fn with positive L2 penalty as in code; printed paper Equation 5 has a minus sign, in conflict with prose/code.',
            'Author Agent.velocities stores dt-scaled displacement. model_input is metres per step; physical velocity_m_s is separately stored.',
            'h0 is linear initial_position_encoder(p0), can be negative; ReLU first occurs at recurrent step 1.',
            'Author overrides Module.train with a custom training loop; fixed-weight inference calls Module.train(model, False) directly.',
            'CPU generation is vectorized from author Agent rules. RNG call ordering and float32 position accumulation differ from serial author trajectories.',
            'Local accelerated recipe differs in learning rate, L2, place layout, number of environments, training budget and PyTorch version.',
            'No strict paper replication, grid-cell replication, torus replication or cross-environment generalization claim.'])
    dump(a/'source_availability.json',report)
    r=ET.parse(a/'paper.xml').getroot()
    titles=['Generating dataset trajectories','The model','Loss function and training details','Data and code availability']
    sections=[]
    for s in r.iter('sec'):
        if s.find('title') is not None and ''.join(s.find('title').itertext()) in titles:
            sections.append(''.join(s.itertext()))
    (a/'paper_methods.txt').write_text('\n\n'.join(sections),encoding='utf-8')
    print(json.dumps({'matching':comparison,'checkpoint_files':found},ensure_ascii=False))
if __name__=='__main__':main()
