from __future__ import annotations
import math, shutil, subprocess, zipfile
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict
import cv2, numpy as np

TARGETS={"1x1":(1080,1080),"4x5":(1080,1350),"16x9":(1920,1080)}
@dataclass
class Box:
    x1:float; y1:float; x2:float; y2:float; kind:str; conf:float=1.; weight:float=1.
    def area(self): return max(0.,self.x2-self.x1)*max(0.,self.y2-self.y1)
    def cx(self): return (self.x1+self.x2)/2
    def cy(self): return (self.y1+self.y2)/2
@dataclass
class FrameAnalysis:
    frame_idx:int; t:float; boxes:List[Box]
@dataclass
class Plan:
    ratio:str; method:str; protection:float; text_protection:float; person_protection:float
    confidence:float; review:bool; reason:str

class Detectors:
    def __init__(self,weights:Dict[str,float],conf=.25):
        self.weights,self.conf=weights,conf; self.ocr=self.yolo=None
        self.hog=cv2.HOGDescriptor(); self.hog.setSVMDetector(cv2.HOGDescriptor_getDefaultPeopleDetector())
        try:
            from paddleocr import PaddleOCR
            self.ocr=PaddleOCR(use_angle_cls=False,lang='en',show_log=False)
        except Exception: pass
        try:
            from ultralytics import YOLO
            self.yolo=YOLO('yolo11n.pt')
        except Exception: pass
    def text_boxes(self,f):
        h,w=f.shape[:2]; out=[]
        if self.ocr:
            try:
                r=self.ocr.ocr(f,det=True,rec=False,cls=False); rows=r[0] if r else []
                for p in rows or []:
                    a=np.array(p,dtype=float).reshape(-1,2); x1,y1=a.min(0); x2,y2=a.max(0)
                    if (x2-x1)*(y2-y1)>w*h*.00015: out.append(Box(x1,y1,x2,y2,'text',1,self.weights['text']))
                if out:return merge(out,max(8,int(w*.008)))
            except Exception: pass
        g=cv2.cvtColor(f,cv2.COLOR_BGR2GRAY); grad=cv2.morphologyEx(g,cv2.MORPH_GRADIENT,np.ones((3,3),np.uint8))
        _,bw=cv2.threshold(grad,0,255,cv2.THRESH_BINARY+cv2.THRESH_OTSU)
        bw=cv2.morphologyEx(bw,cv2.MORPH_CLOSE,cv2.getStructuringElement(cv2.MORPH_RECT,(17,3)),iterations=2)
        cnts,_=cv2.findContours(bw,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
        for c in cnts:
            x,y,ww,hh=cv2.boundingRect(c)
            if ww*hh>w*h*.0002 and ww>24 and hh>8 and ww/max(hh,1)>1.2 and hh<h*.22:
                out.append(Box(x,y,x+ww,y+hh,'text',.55,self.weights['text']))
        return merge(out,max(8,int(w*.008)))
    def object_boxes(self,f):
        out=[]
        if self.yolo:
            try:
                r=self.yolo.predict(f,conf=self.conf,verbose=False)[0]
                for b in r.boxes:
                    c=float(b.conf[0]); name=str(r.names[int(b.cls[0])]); x1,y1,x2,y2=map(float,b.xyxy[0].tolist())
                    k='person' if name=='person' else 'other'; out.append(Box(x1,y1,x2,y2,k,c,self.weights[k]))
                return out
            except Exception: pass
        scale=min(1.,720/f.shape[1]); s=cv2.resize(f,None,fx=scale,fy=scale) if scale<1 else f
        rects,ws=self.hog.detectMultiScale(s,winStride=(8,8),padding=(8,8),scale=1.05)
        for (x,y,w,h),c in zip(rects,ws): out.append(Box(x/scale,y/scale,(x+w)/scale,(y+h)/scale,'person',float(c),self.weights['person']))
        return out
    def detect(self,f): return self.text_boxes(f)+self.object_boxes(f)

def merge(boxes,gap=10):
    out=[]
    for b in sorted(boxes,key=lambda z:(z.y1,z.x1)):
        hit=next((m for m in out if m.kind==b.kind and not(b.y2<m.y1-gap or b.y1>m.y2+gap or b.x2<m.x1-gap or b.x1>m.x2+gap)),None)
        if hit: hit.x1=min(hit.x1,b.x1);hit.y1=min(hit.y1,b.y1);hit.x2=max(hit.x2,b.x2);hit.y2=max(hit.y2,b.y2);hit.conf=max(hit.conf,b.conf)
        else: out.append(Box(**asdict(b)))
    return out

def video_meta(p):
    c=cv2.VideoCapture(str(p)); fps=c.get(cv2.CAP_PROP_FPS) or 30.; n=int(c.get(cv2.CAP_PROP_FRAME_COUNT));w=int(c.get(cv2.CAP_PROP_FRAME_WIDTH));h=int(c.get(cv2.CAP_PROP_FRAME_HEIGHT));c.release()
    return dict(fps=fps,frames=n,width=w,height=h,duration=n/fps)

def analyze_video(p,d,sample_fps=2,max_frames=180,progress=None):
    m=video_meta(p); step=max(1,int(round(m['fps']/sample_fps))); c=cv2.VideoCapture(str(p));out=[];i=0
    while True:
        ok,f=c.read()
        if not ok:break
        if i%step==0:
            out.append(FrameAnalysis(i,i/m['fps'],d.detect(f)))
            if progress:progress(min(1.,i/max(m['frames'],1)))
            if len(out)>=max_frames:break
        i+=1
    c.release();return m,out

def center(bs,w,h):
    if not bs:return w/2,h/2
    q=[b.weight*max(.2,b.conf)*math.sqrt(max(1,b.area())) for b in bs];s=sum(q)
    return sum(b.cx()*z for b,z in zip(bs,q))/s,sum(b.cy()*z for b,z in zip(bs,q))/s

def crop_rect(w,h,ar,cx,cy):
    if w/h>ar: ch=h;cw=h*ar
    else:cw=w;ch=w/ar
    x=float(np.clip(cx-cw/2,0,w-cw));y=float(np.clip(cy-ch/2,0,h-ch));return x,y,x+cw,y+ch

def inter(b,r):
    x1,y1,x2,y2=r; a=max(0,min(b.x2,x2)-max(b.x1,x1))*max(0,min(b.y2,y2)-max(b.y1,y1));return a/max(1,b.area())
def protect(a,r,kind=None):
    n=d=0.
    for f in a:
        for b in f.boxes:
            if kind and b.kind!=kind:continue
            z=b.weight*max(.2,b.conf)*max(1,b.area());d+=z;n+=z*inter(b,r)
    return n/d if d else 1.

def choose_plan(m,a,ratio,cfg):
    tw,th=TARGETS[ratio];w,h=m['width'],m['height'];cs=[center(f.boxes,w,h) for f in a] or [(w/2,h/2)]
    r=crop_rect(w,h,tw/th,float(np.median([x for x,_ in cs])),float(np.median([y for _,y in cs])))
    p,t,pp=protect(a,r),protect(a,r,'text'),protect(a,r,'person');thr=cfg['layout']['crop_min_protection']
    mismatch=max((w/h)/(tw/th),(tw/th)/(w/h))
    if p>=thr and t>=thr: method='Intelligent Crop';reason='중요 영역이 목표 crop 안에 유지됨'
    elif mismatch<=1.38 and min(t,pp)>=.80:method='Scale / Reposition';reason='Crop 손실을 줄이기 위해 원본을 축소·재배치'
    else:method='Background Extend';reason='중요 정보가 넓게 분산되어 배경 확장이 안전함'
    qs=[min(1,max(0,b.conf)) for f in a for b in f.boxes];dq=float(np.mean(qs)) if qs else .45;cov=min(p,t,pp)
    cf=float(np.clip((.58*cov+.42*dq)*{'Intelligent Crop':1,'Scale / Reposition':.94,'Background Extend':.90}[method],0,1))
    rev=cf<cfg['qa']['review_confidence_below'] or cov<cfg['qa']['review_protection_below']
    return Plan(ratio,method,p,t,pp,cf,rev,reason)
def trajectories(m,a,cfg):
    w,h=m['width'],m['height']; pts=[]
    for f in a:
        x,y=center(f.boxes,w,h);pts.append((f.frame_idx,x,y))
    if not pts:pts=[(0,w/2,h/2),(m['frames']-1,w/2,h/2)]
    if pts[-1][0]<m['frames']-1:pts.append((m['frames']-1,pts[-1][1],pts[-1][2]))
    xs=np.interp(np.arange(m['frames']),[p[0] for p in pts],[p[1] for p in pts]);ys=np.interp(np.arange(m['frames']),[p[0] for p in pts],[p[2] for p in pts])
    alpha=cfg['layout']['smoothing_alpha'];limit=cfg['layout']['max_pan_per_frame_ratio']*min(w,h)
    for arr in (xs,ys):
        for i in range(1,len(arr)):
            target=arr[i-1]+alpha*(arr[i]-arr[i-1]);arr[i]=arr[i-1]+np.clip(target-arr[i-1],-limit,limit)
    return xs,ys

def _contain(frame,tw,th,cx,cy,blur=False):
    h,w=frame.shape[:2];scale=min(tw/w,th/h);nw,nh=max(2,int(w*scale)//2*2),max(2,int(h*scale)//2*2)
    fg=cv2.resize(frame,(nw,nh),interpolation=cv2.INTER_AREA)
    if blur:
        bgscale=max(tw/w,th/h);bw,bh=max(2,int(w*bgscale)),max(2,int(h*bgscale));bg=cv2.resize(frame,(bw,bh));x=(bw-tw)//2;y=(bh-th)//2;canvas=cv2.GaussianBlur(bg[y:y+th,x:x+tw],(0,0),35)
    else:canvas=np.zeros((th,tw,3),np.uint8)
    x=int(np.clip(tw/2-nw*(cx/w),0,max(0,tw-nw)));y=int(np.clip(th/2-nh*(cy/h),0,max(0,th-nh)))
    canvas[y:y+nh,x:x+nw]=fg;return canvas

def render_video(src,out,plan,a,m,cfg,bg_mode='auto',progress=None):
    tw,th=TARGETS[plan.ratio];xs,ys=trajectories(m,a,cfg);cap=cv2.VideoCapture(str(src));tmp=Path(str(out)+'.silent.mp4')
    fourcc=cv2.VideoWriter_fourcc(*'mp4v');wr=cv2.VideoWriter(str(tmp),fourcc,m['fps'],(tw,th));i=0
    while True:
        ok,f=cap.read()
        if not ok:break
        cx=xs[min(i,len(xs)-1)];cy=ys[min(i,len(ys)-1)]
        if plan.method=='Intelligent Crop':
            x1,y1,x2,y2=crop_rect(m['width'],m['height'],tw/th,cx,cy);crop=f[int(y1):int(y2),int(x1):int(x2)];o=cv2.resize(crop,(tw,th),interpolation=cv2.INTER_AREA)
        elif plan.method=='Scale / Reposition':o=_contain(f,tw,th,cx,cy,blur=(bg_mode=='blur'))
        else:o=_contain(f,tw,th,cx,cy,blur=(bg_mode in ('auto','blur')))
        wr.write(o);i+=1
        if progress and i%10==0:progress(min(1.,i/max(m['frames'],1)))
    cap.release();wr.release();Path(out).parent.mkdir(parents=True,exist_ok=True)
    cmd=['ffmpeg','-y','-i',str(tmp),'-i',str(src),'-map','0:v:0','-map','1:a?','-c:v','libx264','-crf',str(cfg['output']['video_crf']),'-preset',cfg['output']['preset'],'-pix_fmt','yuv420p','-c:a','aac','-b:a',cfg['output']['audio_bitrate'],'-shortest','-movflags','+faststart',str(out)]
    r=subprocess.run(cmd,stdout=subprocess.PIPE,stderr=subprocess.PIPE)
    tmp.unlink(missing_ok=True)
    if r.returncode:raise RuntimeError(r.stderr.decode(errors='ignore')[-3000:])
    return out

def make_zip(files,out):
    with zipfile.ZipFile(out,'w',zipfile.ZIP_DEFLATED) as z:
        for p in files:z.write(p,arcname=str(Path(p).relative_to(Path(p).parents[2])) if len(Path(p).parents)>2 else Path(p).name)
    return out
