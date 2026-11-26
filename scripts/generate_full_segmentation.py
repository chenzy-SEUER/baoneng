"""全量 train/val 图像引导分割伪标签；保留已审核轮廓及原划分。"""
import argparse
import copy
import hashlib
import json
import shutil
from collections import Counter
from pathlib import Path

import cv2
import numpy as np
from PIL import Image

from scripts.prepare_segmentation_pilot import ROOT, read_json, write_json, validate_polygons

def proposal(rgb,box):
 h,w=rgb.shape[:2];x1,y1,x2,y2=[int(v) for v in box]
 x1=max(0,x1);y1=max(0,y1);x2=min(w,x2);y2=min(h,y2)
 margin=max(8,min(20,round(max(x2-x1,y2-y1)*.3)))
 a,b=max(0,x1-margin),max(0,y1-margin);c,d=min(w,x2+margin),min(h,y2+margin)
 roi=rgb[b:d,a:c].copy();mask=np.full(roi.shape[:2],cv2.GC_BGD,np.uint8)
 sx,sy=x1-a,y1-b;ex,ey=x2-a,y2-b
 mask[sy:ey,sx:ex]=cv2.GC_PR_FGD
 lab=cv2.cvtColor(roi,cv2.COLOR_RGB2LAB)[:,:,0].astype(float)
 inside=lab[sy:ey,sx:ex]
 # High-response seed pixels; excludes margin. No fabricated shape fallback.
 seed=(inside>=np.percentile(inside,90))
 mask[sy:ey,sx:ex][seed]=cv2.GC_FGD
 bg=np.zeros((1,65),np.float64);fg=np.zeros((1,65),np.float64)
 flags=[]
 try:cv2.grabCut(cv2.cvtColor(roi,cv2.COLOR_RGB2BGR),mask,None,bg,fg,5,cv2.GC_INIT_WITH_MASK)
 except cv2.error:
  mask[:]=cv2.GC_BGD
  mask[sy:ey,sx:ex][seed]=cv2.GC_FGD
  flags.append('grabcut_failed_seed_only_review')
 binary=np.uint8((mask==cv2.GC_FGD)|(mask==cv2.GC_PR_FGD))*255
 binary[:sy,:]=0;binary[ey:,:]=0;binary[:,:sx]=0;binary[:,ex:]=0
 binary=cv2.morphologyEx(binary,cv2.MORPH_CLOSE,np.ones((3,3),np.uint8))
 # Retain the reference support restriction after morphological closing.
 support=np.zeros_like(binary);support[sy:ey,sx:ex]=255;binary=cv2.bitwise_and(binary,support)
 count,components,stats,_=cv2.connectedComponentsWithStats(binary)
 parts=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=2]
 if not parts:
  parts=[i for i in range(1,count) if stats[i,cv2.CC_STAT_AREA]>=1]
  flags.append('single_pixel_components_review')
 if not parts:return None,['no_foreground']
 retained=np.uint8(np.isin(components,parts))*255
 contours,_=cv2.findContours(retained,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 if len(contours)>1:
  # This envelope is only a review proposal: no claim that all bridge pixels are affected.
  contour=cv2.convexHull(np.concatenate(contours));flags.append('multi_component_envelope_review')
 else:contour=contours[0]
 area=cv2.contourArea(contour)
 if area<1:
  enlarged=cv2.resize(retained,None,fx=3,fy=3,interpolation=cv2.INTER_NEAREST)
  tiny,_=cv2.findContours(enlarged,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
  contour=cv2.convexHull(np.concatenate(tiny)).reshape(-1,2).astype(float)/3
  flags.append('subpixel_contour_review')
 else:
  simplified=cv2.approxPolyDP(contour,.65,True).reshape(-1,2)
  contour=simplified if len(simplified)>=3 else contour.reshape(-1,2)
 if len(contour)<3:return None,['degenerate_contour']
 points=[[round(float(x+a),4),round(float(y+b),4)] for x,y in contour]
 fraction=area/max(1,(x2-x1)*(y2-y1))
 if fraction<.2:flags.append('small_foreground_review')
 if fraction>.9:flags.append('nearly_reference_box_review')
 xs=[p[0] for p in points];ys=[p[1] for p in points]
 if max(xs)-min(xs)<.65*(x2-x1) or max(ys)-min(ys)<.65*(y2-y1):flags.append('weak_extent_may_be_missing')
 if min(x1,y1)==0 or x2==w or y2==h:flags.append('image_edge')
 flags.append('automatic_boundary_review_required')
 return points,flags

def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--manifest', type=Path, default=ROOT/'output/dataset/manifest.json')
    parser.add_argument('--approved-package', type=Path, default=ROOT/'output/segmentation_polygon_batch_20261126')
    parser.add_argument('--out-dir', type=Path, required=True)
    args = parser.parse_args()
    original = read_json(args.manifest)
    records = [r for r in original if r['split'] in {'train','val'}]
    ids = [r['id'] for r in records]
    if len(ids) != len(set(ids)):
        raise ValueError('重复ID')
    for r in records:
        path=Path(r['image_path']);path=path if path.is_absolute() else ROOT/path
        if not path.is_file(): raise FileNotFoundError(path)
    args.out_dir.mkdir(parents=True,exist_ok=False)
    cv2.setRNGSeed(42);cv2.setNumThreads(1)
    review=[];failures=[];approved_images=0
    old_index_path=args.approved_package/'review_index.json'
    old_index={(x['id'],x['object_index']):x for x in read_json(old_index_path)} if old_index_path.is_file() else {}
    for number,r in enumerate(records,1):
        folder=args.out_dir/'annotations'/r['split'];folder.mkdir(parents=True,exist_ok=True)
        src=Path(r['image_path']);src=src if src.is_absolute() else ROOT/src
        filename=r['id']+src.suffix
        with Image.open(src) as im:
            image=im.convert('RGB');rgb=np.array(image)
        if image.size!=(r['width'],r['height']):raise ValueError('图像尺寸变化：'+r['id'])
        approved=args.approved_package/'annotations'/r['split']/(r['id']+'.json')
        if approved.is_file():
            ann=read_json(approved)
            if ann.get('flags',{}).get('reviewed') is not True:raise ValueError('旧标注尚未审核：'+r['id'])
            validate_polygons(ann,r)
            previous_image=approved.parent/ann['imagePath']
            if hashlib.sha256(previous_image.read_bytes()).digest()!=hashlib.sha256(src.read_bytes()).digest():raise ValueError('旧标注对应图像变化')
            ann=copy.deepcopy(ann);approved_images+=1
            polys={int(s['label'][4:]):s['points'] for s in ann['shapes'] if s['shape_type']=='polygon'}
            for i,obj in enumerate(r['objects']):
                review.append({'id':r['id'],'split':r['split'],'object_index':i,'class':obj['label'],
                    'reviewed':True,'source':'previously_user_reviewed','polygon':polys[i],
                    'review_flags':old_index.get((r['id'],i),{}).get('review_flags',[])})
        else:
            ann={'version':'5.0.0','flags':{'reviewed':False,'draft':True},'shapes':[],
                 'imagePath':filename,'imageData':None,'imageHeight':r['height'],'imageWidth':r['width']}
            for i,obj in enumerate(r['objects']):
                x1,y1,x2,y2=obj['box']
                ann['shapes'].append({'label':f"reference_{i}_{obj['label']}",'points':[[x1,y1],[x2,y2]],
                                     'group_id':None,'shape_type':'rectangle','flags':{}})
                points,flags=proposal(rgb,obj['box'])
                if points:
                    ann['shapes'].append({'label':f'obj_{i}','points':points,'group_id':None,'shape_type':'polygon',
                        'flags':{'draft':True},'description':'GrabCut image-guided pseudo-label, reference-box constrained; weak boundaries and connected envelopes not manually verified.'})
                else:failures.append({'id':r['id'],'object_index':i,'flags':flags})
                review.append({'id':r['id'],'split':r['split'],'object_index':i,'class':obj['label'],
                    'reviewed':False,'source':'automatic_image_guided','polygon':points,'review_flags':flags})
            if all(s['polygon'] is not None for s in review[-len(r['objects']):]):
                validate_polygons(ann,r,allow_drafts=True)
        ann['imagePath']=filename
        write_json(folder/(r['id']+'.json'),ann);shutil.copy2(src,folder/filename)
        if number%100==0 or number==len(records):
            print(f'{number}/{len(records)} images; polygons={len(review)}; failures={len(failures)}',flush=True)
    write_json(args.out_dir/'manifest.json',records)
    write_json(args.out_dir/'review_index.json',review)
    write_json(args.out_dir/'failures.json',failures)
    write_json(args.out_dir/'held_out_test_ids.json',[r['id'] for r in original if r['split']=='test'])
    summary={'images_by_split':dict(Counter(r['split'] for r in records)),
             'instances_by_split':dict(Counter(r['split'] for r in records for _ in r['objects'])),
             'class_counts':dict(Counter(x['class'] for x in review)),
             'approved_images_reused':approved_images,'approved_polygons_reused':sum(x['reviewed'] for x in review),
             'automatic_polygons':sum(not x['reviewed'] for x in review),'failed_polygons':len(failures),
             'review_flags':dict(Counter(f for x in review if not x['reviewed'] for f in x['review_flags'])),
             'manifest_sha256':hashlib.sha256(args.manifest.read_bytes()).hexdigest()}
    write_json(args.out_dir/'generation_summary.json',summary)
    print(json.dumps(summary,ensure_ascii=False,indent=2),flush=True)
    if failures:raise ValueError('部分轮廓失败，详见failures.json；未允许跳过实例导出')


if __name__=='__main__':
    main()
