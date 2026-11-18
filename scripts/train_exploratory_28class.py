import argparse, csv, json, os, random, time
from pathlib import Path
import numpy as np
from PIL import Image
from sklearn.model_selection import train_test_split
import torch
from torch import nn
from torch.utils.data import Dataset, DataLoader
from torchvision import models, transforms

ROOT = Path(__file__).resolve().parents[1]
RUN = ROOT / 'runs' / 'exploratory_28class_seed42'
SOURCE = ROOT / 'data/PlantVillage/raw/color'
SEED = 42

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--benchmark', action='store_true')
    args = parser.parse_args()
    random.seed(SEED); np.random.seed(SEED); torch.manual_seed(SEED); torch.cuda.manual_seed_all(SEED)
    RUN.mkdir(parents=True, exist_ok=True)
    with (RUN / 'candidate_alignment.csv').open(encoding='utf-8-sig', newline='') as f:
        classes = sorted(csv.DictReader(f), key=lambda x: x['plantdoc_folder'].casefold())
    assert len(classes) == 28 and len({c['plantvillage_folder'] for c in classes}) == 28
    samples = []
    for idx, cls in enumerate(classes):
        directory = SOURCE / cls['plantvillage_folder']
        if not directory.is_dir(): raise RuntimeError(f'Missing class: {directory}')
        paths = sorted(p for p in directory.rglob('*') if p.is_file() and p.suffix.lower() in {'.jpg','.jpeg','.png'})
        if not paths: raise RuntimeError(f'Empty class: {directory}')
        samples.extend((str(p), idx) for p in paths)
    train, rest = train_test_split(samples, test_size=.30, random_state=SEED, stratify=[x[1] for x in samples])
    val, test = train_test_split(rest, test_size=.5, random_state=SEED, stratify=[x[1] for x in rest])
    for name, rows in [('train', train), ('val', val), ('test', test)]:
        path = RUN / f'source_{name}.csv'
        if not path.exists():
            with path.open('w',newline='',encoding='utf-8') as f:
                w=csv.writer(f); w.writerow(['path','class_index']); w.writerows(rows)
        else:
            with path.open(newline='',encoding='utf-8') as f:
                existing = list(csv.reader(f))[1:]
            if existing != [[p,str(i)] for p,i in rows]: raise RuntimeError(f'Manifest mismatch: {path}')
    config = {'status':'exploratory_28_class_not_paper_21_class','seed':SEED,'counts':{'train':len(train),'val':len(val),'test':len(test)},'class_order':classes,'optimizer':'AdamW lr=0.0001 weight_decay=0.01','batch_size':32,'max_epochs':20,'early_stopping_patience':5,'pretrained':'torchvision ResNet50_Weights.IMAGENET1K_V1','augmentation_assumptions':'resize 224; random resized crop scale 0.8-1.0; horizontal and vertical flips; color jitter 0.2,0.2,0.2,0.05; ImageNet normalization','eval_assumptions':'resize 256 center crop 224; ImageNet normalization','warning':'Candidate mappings unverified. Seven unknown classes and paper split unavailable. Do not compare scores directly to paper.'}
    (RUN/'config.json').write_text(json.dumps(config,ensure_ascii=False,indent=2),encoding='utf-8')
    print('Counts:',config['counts'], flush=True)
    norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225])
    aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm])
    ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
    class Images(Dataset):
        def __init__(self,rows,tf): self.rows=rows; self.tf=tf
        def __len__(self): return len(self.rows)
        def __getitem__(self,i):
            p,y=self.rows[i]
            with Image.open(p) as im: return self.tf(im.convert('RGB')),y
    tr=DataLoader(Images(train,aug),batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
    va=DataLoader(Images(val,ev),batch_size=32,shuffle=False,num_workers=4,pin_memory=True)
    model=models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V1)
    model.fc=nn.Linear(model.fc.in_features,len(classes)); model.cuda()
    opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01)
    sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=20)
    lossfn=nn.CrossEntropyLoss()
    if args.benchmark:
        model.train(); start=time.time(); n=0
        for x,y in tr:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
            loss=lossfn(model(x),y); loss.backward(); opt.step(); n+=len(y)
            if n>=3200: break
        torch.cuda.synchronize(); seconds=time.time()-start
        print(f'BENCHMARK {n} images {seconds:.1f}s {n/seconds:.2f} images/s; training-only epoch estimate {len(train)/(n/seconds)/60:.1f} min',flush=True)
        return
    if (RUN/'history.csv').exists() or (RUN/'best.pt').exists(): raise RuntimeError('Existing run artifacts: refusing overwrite')
    best=-1; stale=0
    for epoch in range(1,21):
        start=time.time(); model.train(); total=0.; n=0
        for x,y in tr:
            x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True); opt.zero_grad(set_to_none=True)
            loss=lossfn(model(x),y); loss.backward(); opt.step(); total+=loss.item()*len(y); n+=len(y)
        sched.step(); model.eval(); correct=0; seen=0
        with torch.no_grad():
            for x,y in va:
                x=x.cuda(non_blocking=True); y=y.cuda(non_blocking=True)
                correct+=(model(x).argmax(1)==y).sum().item(); seen+=len(y)
        acc=correct/seen; elapsed=time.time()-start
        improved=acc>best
        if improved:
            best=acc; stale=0
            torch.save({'model':model.state_dict(),'epoch':epoch,'val_accuracy':acc,'config':config},RUN/'best.pt')
        else: stale+=1
        row={'epoch':epoch,'train_loss':total/n,'val_accuracy':acc,'seconds':elapsed,'best_val_accuracy':best,'stale_epochs':stale}
        with (RUN/'history.csv').open('a',newline='',encoding='utf-8') as f:
            w=csv.DictWriter(f,fieldnames=row.keys())
            if epoch==1: w.writeheader()
            w.writerow(row)
        print(json.dumps(row),flush=True)
        if stale>=5: print('Early stopping',flush=True); break
    (RUN/'TRAINING_COMPLETE').write_text('completed\n')
if __name__=='__main__': main()
