import csv, json, hashlib, shutil
from pathlib import Path
from collections import defaultdict, Counter
from PIL import Image
from sklearn.model_selection import train_test_split
R=Path(__file__).resolve().parents[1]; O=R/'data/PlantDoc-Original'; B=R/'runs/three_routes_v1'; B.mkdir(parents=True,exist_ok=True)
C=R/'runs/exploratory_28class_seed42/candidate_alignment.csv'
with C.open(encoding='utf-8-sig',newline='') as f: classes=sorted(csv.DictReader(f),key=lambda x:x['plantdoc_folder'].casefold())
lookup={x['plantdoc_folder']:i for i,x in enumerate(classes)}
annotations={}; issues=Counter()
for split in ['train','test']:
 with (O/f'{split}_labels.csv').open(encoding='utf-8-sig',newline='') as f:
  for row in csv.DictReader(f): annotations.setdefault((split,row['filename']),[]).append(row)
images=[]
for (split,name),rows in annotations.items():
 path=O/split.upper()/name
 if not path.is_file(): issues['missing_images']+=1; continue
 try:
  with Image.open(path) as im: w,h=im.size; im.verify()
 except Exception: issues['unreadable_images']+=1; continue
 digest=hashlib.sha256(path.read_bytes()).hexdigest()
 boxes=[]
 for j,row in enumerate(rows):
  try:
   x1,y1,x2,y2=[int(float(row[k])) for k in ('xmin','ymin','xmax','ymax')]
   if (int(row['width']),int(row['height'])) != (w,h): issues['annotation_dimension_mismatch']+=1; continue
   x1=max(0,min(x1,w)); x2=max(0,min(x2,w)); y1=max(0,min(y1,h)); y2=max(0,min(y2,h))
   if x2<=x1 or y2<=y1: issues['invalid_boxes']+=1; continue
  except Exception: issues['invalid_boxes']+=1; continue
  boxes.append({'box':[x1,y1,x2,y2],'class':row['class'],'class_index':lookup.get(row['class'],-1),'annotation_index':j})
 if not boxes: issues['no_valid_boxes']+=1; continue
 images.append({'original_split':split,'filename':name,'path':str(path),'sha256':digest,'width':w,'height':h,'boxes':boxes})
testhash={x['sha256'] for x in images if x['original_split']=='test'}
testnames={x['filename'].casefold() for x in images if x['original_split']=='test'}
train=[x for x in images if x['original_split']=='train' and x['sha256'] not in testhash and x['filename'].casefold() not in testnames]
issues['train_images_excluded_test_overlap']=sum(1 for x in images if x['original_split']=='train')-len(train)
test=[x for x in images if x['original_split']=='test']
# Group byte-identical images so duplicates stay in one partition.
groups=sorted({x['sha256'] for x in train}); tr,va=train_test_split(groups,test_size=.15,random_state=42)
sets={'train':set(tr),'val':set(va)}
for x in images:
 x['partition']='test' if x in test else next((k for k,v in sets.items() if x['sha256'] in v),'excluded')
# Stable leaf and image ids retain all valid annotations, with unknown categories left in detection.
for p in ['train','val','test']:
 for sub in ['images','labels']:(B/'yolo'/sub/p).mkdir(parents=True,exist_ok=True)
records=[]
for i,x in enumerate(images):
 x['image_id']=f'{i:05d}'; part=x['partition']
 if part=='excluded':continue
 ext=Path(x['filename']).suffix.lower() or '.jpg'; dest=B/'yolo/images'/part/(x['image_id']+ext)
 if not dest.exists():
  try:
   dest.symlink_to(x['path'])
  except OSError:
   shutil.copyfile(x['path'], dest)
 lines=[]
 for box in x['boxes']:
  x1,y1,x2,y2=box['box']; w=x['width']; h=x['height']
  lines.append(f'0 {(x1+x2)/(2*w):.8f} {(y1+y2)/(2*h):.8f} {(x2-x1)/w:.8f} {(y2-y1)/h:.8f}')
  if box['class_index']>=0: records.append({'image_id':x['image_id'],'partition':part,'image_path':x['path'],'sha256':x['sha256'],'class':box['class'],'class_index':box['class_index'],'box':box['box'],'annotation_index':box['annotation_index']})
 (B/'yolo/labels'/part/(x['image_id']+'.txt')).write_text('\n'.join(lines)+'\n')
(B/'images.json').write_text(json.dumps(images,ensure_ascii=False,indent=2),encoding='utf-8')
(B/'leaves.json').write_text(json.dumps(records,ensure_ascii=False,indent=2),encoding='utf-8')
(B/'yolo/dataset.yaml').write_text(f'path: {json.dumps((B / "yolo").as_posix(), ensure_ascii=False)}\ntrain: images/train\nval: images/val\ntest: images/test\nnames:\n  0: leaf\n')
summary={'images':dict(Counter(x['partition'] for x in images)),'known_leaf_boxes':dict(Counter(x['partition'] for x in records)),'issues':dict(issues),'classes':classes,'protocol':'original dataset test held out; remove exact image and filename overlaps; split training by exact image hash seed42; 28 unverified candidate classes; single-class leaf detection; IoU>=0.5 matching; source-only and mixed classifiers use identical known test boxes; detection misses count incorrect; unknown classes retained for detection, excluded from classification; label dimensions must match actual image; no claim of paper exact split'}
(B/'protocol.json').write_text(json.dumps(summary,ensure_ascii=False,indent=2),encoding='utf-8')
print(json.dumps({k:v for k,v in summary.items() if k!='classes'},indent=2),flush=True)
