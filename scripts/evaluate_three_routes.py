import csv,json,math
from pathlib import Path
from collections import defaultdict
import numpy as np
from PIL import Image
import torch
from torch import nn
from torch.utils.data import Dataset,DataLoader
from torchvision import models,transforms
from sklearn.metrics import accuracy_score,f1_score,classification_report
from scipy.optimize import linear_sum_assignment
R=Path(__file__).resolve().parents[1]; B=R/'runs/three_routes_v1'; P=R/'runs/exploratory_28class_seed42'; D=R/'data/PlantDoc-Cropped'
classes=json.loads((B/'protocol.json').read_text(encoding='utf-8'))['classes']; names=[x['plantdoc_folder'] for x in classes]; label={n:i for i,n in enumerate(names)}
T=transforms.Compose([transforms.Resize(256),transforms.CenterCrop(224),transforms.ToTensor(),transforms.Normalize([.485,.456,.406],[.229,.224,.225])])
class Data(Dataset):
 def __init__(self,rows): self.rows=rows
 def __len__(self):return len(self.rows)
 def __getitem__(self,i):
  r=self.rows[i]
  with Image.open(r['path']) as im:
   im=im.convert('RGB'); im=im.crop(tuple(r['box'])) if r.get('box') else im
   return T(im),r['y'],i

def evaluate(model,rows,name):
 dl=DataLoader(Data(rows),batch_size=48,num_workers=4,pin_memory=True); results=[]
 model.eval()
 with torch.no_grad():
  for x,y,idx in dl:
   prob=model(x.cuda(non_blocking=True)).softmax(1).cpu(); conf,pred=prob.max(1)
   for i,t,p,c in zip(idx.tolist(),y.tolist(),pred.tolist(),conf.tolist()):
    r=rows[i];results.append({'path':r['path'],'box':json.dumps(r.get('box')),'label':names[t],'predicted':names[p],'correct':int(t==p),'confidence':c,'image_id':r.get('image_id','')})
 with (B/(name+'_predictions.csv')).open('w',encoding='utf-8',newline='') as f:
  w=csv.DictWriter(f,fieldnames=results[0].keys());w.writeheader();w.writerows(results)
 y=[names.index(x['label']) for x in results]; pred=[names.index(x['predicted']) for x in results]
 summary={'n':len(y),'correct':sum(x['correct'] for x in results),'accuracy':accuracy_score(y,pred),'macro_f1_over_present_classes':f1_score(y,pred,labels=sorted(set(y)),average='macro',zero_division=0),'mean_confidence':float(np.mean([x['confidence'] for x in results])),'per_class':classification_report(y,pred,labels=list(range(28)),target_names=names,output_dict=True,zero_division=0)}
 (B/(name+'_metrics.json')).write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
 print(name,{k:v for k,v in summary.items() if k!='per_class'},flush=True)
 return results

def get_model(path):
 m=models.resnet50(weights=None);m.fc=nn.Linear(m.fc.in_features,28);m.load_state_dict(torch.load(path,map_location='cpu',weights_only=False)['model']);return m.cuda().eval()

def iou(a,b):
 x1=max(a[0],b[0]);y1=max(a[1],b[1]);x2=min(a[2],b[2]);y2=min(a[3],b[3]);inter=max(0,x2-x1)*max(0,y2-y1);ua=(a[2]-a[0])*(a[3]-a[1])+(b[2]-b[0])*(b[3]-b[1])-inter;return inter/ua if ua>0 else 0

def main():
 assert (B/'MIXED_COMPLETE').exists() and (B/'YOLO_COMPLETE').exists(), 'Finish all training before final test'
 leaves=json.loads((B/'leaves.json').read_text(encoding='utf-8')); target=[{'path':x['image_path'],'box':x['box'],'y':x['class_index'],'image_id':x['image_id']} for x in leaves if x['partition']=='test']
 with (P/'source_test.csv').open(newline='',encoding='utf-8') as f: source=[{'path':r['path'],'box':None,'y':int(r['class_index'])} for r in csv.DictReader(f)]
 cropped=[]
 for split in ['train','test']:
  for cls,i in label.items():
   folder=D/split/cls
   if folder.exists():
    for p in sorted(folder.iterdir()):
     if p.is_file() and p.suffix.lower() in ['.jpg','.jpeg','.png']:cropped.append({'path':str(p),'box':None,'y':i})
 print('frozen evaluation counts:',len(source),len(cropped),len(target),flush=True)
 base=get_model(P/'best.pt'); evaluate(base,source,'source_test');evaluate(base,cropped,'public_cropped_all');evaluate(base,target,'common_groundtruth_crop_source_only')
 mixed=get_model(B/'mixed_best.pt');evaluate(mixed,target,'common_groundtruth_crop_mixed')
 from ultralytics import YOLO
 detector=YOLO(str(B/'leaf_detector/weights/best.pt'))
 by_image=defaultdict(list)
 for x in target:by_image[x['image_id']].append(x)
 matching=[]; total_pred=0
 for image_id,gt in sorted(by_image.items()):
  result=detector.predict(source=gt[0]['path'],imgsz=640,conf=.25,iou=.7,device=0,verbose=False,max_det=300)[0]
  boxes=result.boxes.xyxy.cpu().numpy().tolist();total_pred+=len(boxes)
  overlaps=np.array([[iou(g['box'],b) for b in boxes] for g in gt]) if boxes else np.zeros((len(gt),0))
  pairs={}
  if boxes:
   weights=np.where(overlaps>=.5,1+overlaps*.001,0); ri,ci=linear_sum_assignment(weights,maximize=True)
   pairs={int(r):int(c) for r,c in zip(ri,ci) if overlaps[r,c]>=.5}
  for i,g in enumerate(gt):
   p=pairs.get(i);matching.append({'path':g['path'],'box':boxes[p] if p is not None else None,'y':g['y'],'image_id':image_id,'matched_iou':float(overlaps[i,p]) if p is not None else None})
 matched=[x for x in matching if x['box'] is not None]; preds=evaluate(base,matched,'yolo_detected_source_only') if matched else []
 score={'total_ground_truth':len(target),'predicted_boxes':total_pred,'matched':len(matched),'detection_recall_at_iou_0_5':len(matched)/len(target),'classification_accuracy_given_match':sum(x['correct'] for x in preds)/len(matched) if matched else None,'end_to_end_accuracy_misses_incorrect':sum(x['correct'] for x in preds)/len(target),'correct':sum(x['correct'] for x in preds),'detector_confidence_threshold':.25,'nms_iou':.7,'matching_iou':.5,'caveat':'false positive detections are not counted in per-ground-truth accuracy; detection precision is separately reflected by matched/predicted counts'}
 (B/'yolo_pipeline_metrics.json').write_text(json.dumps(score,indent=2),encoding='utf-8');print('yolo_pipeline',score,flush=True)
 (B/'EVALUATION_COMPLETE').write_text('completed\n')
if __name__=='__main__':main()
