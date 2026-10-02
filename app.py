from pathlib import Path
import tempfile, yaml, streamlit as st
from engine import Detectors, analyze_video, choose_plan, render_video, make_zip, TARGETS

st.set_page_config(page_title='VANCED Auto Resize',layout='wide')
st.title('VANCED · 광고 영상 AI Auto Resize V1')
st.caption('Burn-in 텍스트와 인물 Safe Area를 보호하며 광고 규격별 레이아웃을 자동 선택합니다.')
with open('config.yaml','r',encoding='utf-8') as f: cfg=yaml.safe_load(f)
with st.sidebar:
    st.subheader('중요도 Weight')
    for k,label in [('text','텍스트'),('product','제품'),('person','얼굴/인물'),('other','기타 객체')]:cfg['weights'][k]=st.slider(label,0,60,int(cfg['weights'][k]))
    bg=st.radio('배경 방식',['자동','Blur','단색']); bg_mode={'자동':'auto','Blur':'blur','단색':'solid'}[bg]
files=st.file_uploader('MP4 업로드 (Batch 가능)',type=['mp4'],accept_multiple_files=True)
ratios=st.multiselect('출력 비율',list(TARGETS),default=['1x1','4x5'])
if st.button('자동 변환',type='primary',disabled=not(files and ratios)):
    root=Path(tempfile.mkdtemp(prefix='vanced_resize_')); outputs=[]; rows=[]
    det=Detectors(cfg['weights'],cfg['analysis']['detection_confidence'])
    for fi,u in enumerate(files):
        src=root/u.name;src.write_bytes(u.getbuffer());st.subheader(f'{fi+1}. {u.name}')
        p=st.progress(0,text='영상 분석 중…');meta,analysis=analyze_video(src,det,cfg['analysis']['sample_fps'],cfg['analysis']['max_analysis_frames'],lambda x:p.progress(x,text='영상 분석 중…'));p.empty()
        cols=st.columns(len(ratios))
        for col,ratio in zip(cols,ratios):
            plan=choose_plan(meta,analysis,ratio,cfg);out=root/'output'/ratio/f'{src.stem}_{ratio}.mp4';q=col.progress(0,text=f'{ratio} 렌더링…')
            render_video(src,out,plan,analysis,meta,cfg,bg_mode,lambda x:q.progress(x,text=f'{ratio} 렌더링…'));q.empty();outputs.append(out)
            with col:
                st.markdown(f'### {ratio} · {plan.method}')
                st.video(str(out));st.write(plan.reason)
                st.metric('자동 변환 신뢰도',f'{plan.confidence*100:.1f}%');st.write(f'텍스트 보호율 **{plan.text_protection*100:.1f}%** · 인물 보호율 **{plan.person_protection*100:.1f}%**')
                if plan.review:st.warning('자동 변환 검토 권장')
                else:st.success('자동 QA 기준 통과')
                st.download_button(f'{ratio} MP4 다운로드',out.read_bytes(),file_name=out.name,mime='video/mp4')
            rows.append({'file':u.name,'ratio':ratio,'method':plan.method,'text_protection':plan.text_protection,'person_protection':plan.person_protection,'confidence':plan.confidence,'review':plan.review})
    z=root/'vanced_auto_resize_outputs.zip';make_zip(outputs,z)
    st.divider();st.download_button('전체 결과 ZIP 다운로드',z.read_bytes(),file_name=z.name,mime='application/zip',type='primary');st.dataframe(rows,use_container_width=True)
