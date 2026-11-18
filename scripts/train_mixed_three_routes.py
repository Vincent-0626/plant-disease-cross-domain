import csv,json,random,time
from pathlib import Path
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
R=Path(__file__).resolve().parents[1]; B=R/'runs/three_routes_v1'; P=R/'runs/exploratory_28class_seed42'; random.seed(42);np.random.seed(42);torch.manual_seed(42);torch.cuda.manual_seed_all(42)
leaves=json.loads((B/'leaves.json').read_text(encoding='utf-8')); tr=[x for x in leaves if x['partition']=='train']; val=[x for x in leaves if x['partition']=='val']
with (P/'source_train.csv').open(newline='',encoding='utf-8') as f: src=list(csv.DictReader(f))
train=[{'image_path':r['path'],'class_index':int(r['class_index']),'box':None} for r in src]+tr
norm=transforms.Normalize([.485,.456,.406],[.229,.224,.225]); aug=transforms.Compose([transforms.Resize((224,224)),transforms.RandomResizedCrop(224,scale=(.8,1.)),transforms.RandomHorizontalFlip(),transforms.RandomVerticalFlip(),transforms.ColorJitter(.2,.2,.2,.05),transforms.ToTensor(),norm]); ev=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),norm])
class Images(Dataset):
 def __init__(self,rows,t): self.rows=rows;self.t=t
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  row=self.rows[i]
  with Image.open(row['image_path']) as im:
   im=im.convert('RGB'); box=row.get('box'); im=im.crop(tuple(box)) if box else im; return self.t(im),row['class_index']
train_dl=DataLoader(Images(train,aug),batch_size=32,shuffle=True,num_workers=4,pin_memory=True)
val_dl=DataLoader(Images(val,ev),batch_size=32,num_workers=4,pin_memory=True)
model=models.resnet50(weights=None);model.fc=nn.Linear(model.fc.in_features,28);model.load_state_dict(torch.load(P/'best.pt',map_location='cpu',weights_only=False)['model']);model.cuda()
opt=torch.optim.AdamW(model.parameters(),lr=.0001,weight_decay=.01); sched=torch.optim.lr_scheduler.CosineAnnealingLR(opt,T_max=12);lossfn=nn.CrossEntropyLoss();best=-1;stale=0
if (B/'mixed_history.csv').exists(): raise RuntimeError('Existing mixed training log')
print(f'source training images {len(src)}; target boxes {len(tr)}; field validation boxes {len(val)}',flush=True)
for epoch in range(1,13):
 start=time.time();model.train();total=0;n=0
 for x,y in train_dl:
  x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);opt.zero_grad(set_to_none=True);loss=lossfn(model(x),y);loss.backward();opt.step();total+=loss.item()*len(y);n+=len(y)
 sched.step();model.eval();correct=0;seen=0
 with torch.no_grad():
  for x,y in val_dl:
   x=x.cuda(non_blocking=True);y=y.cuda(non_blocking=True);correct+=(model(x).argmax(1)==y).sum().item();seen+=len(y)
 acc=correct/seen
 if acc>best:
  best=acc;stale=0;torch.save({'model':model.state_dict(),'epoch':epoch,'field_val_accuracy':acc},B/'mixed_best.pt')
 else:stale+=1
 row={'epoch':epoch,'loss':total/n,'field_val_accuracy':acc,'best':best,'seconds':time.time()-start,'stale':stale}
 with (B/'mixed_history.csv').open('a',newline='') as f:
  w=csv.DictWriter(f,fieldnames=row.keys());
  if epoch==1:w.writeheader()
  w.writerow(row)
 print(json.dumps(row),flush=True)
 if stale>=4:break
(B/'MIXED_COMPLETE').write_text('completed\n')
