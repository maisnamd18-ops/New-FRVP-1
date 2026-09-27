import time
from datetime import datetime, timedelta, timezone
from zoneinfo import ZoneInfo
import numpy as np
import pandas as pd
import requests
import streamlit as st
import plotly.graph_objects as go

st.set_page_config(page_title='FRVP POC Delta V13 Backtester', page_icon='📊', layout='wide')
BASE='https://api.india.delta.exchange'; IST=ZoneInfo('Asia/Kolkata')

@st.cache_data(ttl=300)
def products():
    r=requests.get(BASE+'/v2/products',params={'states':'live','page_size':100},timeout=20); r.raise_for_status()
    return r.json().get('result',[])

def gold_symbols(ps):
    out=[]
    for p in ps:
        s=str(p.get('symbol','')); t=' '.join(str(p.get(k,'')) for k in ['symbol','description','underlying_asset_symbol','contract_type']).upper()
        if 'XAU' in t or 'GOLD' in t: out.append(s)
    return sorted(set(out))

@st.cache_data(ttl=300)
def candles(symbol,res,start,end):
    sec={'1m':60,'15m':900}[res]; step=sec*1990; rows=[]; cur=int(start)
    while cur<int(end):
        e=min(int(end),cur+step)
        r=requests.get(BASE+'/v2/history/candles',params={'resolution':res,'symbol':symbol,'start':cur,'end':e},timeout=30); r.raise_for_status()
        j=r.json();
        if not j.get('success'): raise RuntimeError(j)
        rows += j.get('result',[]); cur=e+sec; time.sleep(.05)
    if not rows:return pd.DataFrame()
    d=pd.DataFrame(rows); cols=['time','open','high','low','close','volume']
    if not all(c in d for c in cols): raise RuntimeError(f'Unexpected columns: {list(d)}')
    d=d[cols].drop_duplicates('time').sort_values('time'); d['time']=pd.to_datetime(d.time,unit='s',utc=True); d=d.set_index('time')
    for c in cols[1:]:d[c]=pd.to_numeric(d[c],errors='coerce')
    return d.dropna(subset=['open','high','low','close'])

def sess(ts): return ts.astimezone(IST).date()

def frvp(d,rows=60,va=70):
    lo=float(d.low.min()); hi=float(d.high.max())
    if not np.isfinite(lo+hi) or hi<=lo:return np.nan,np.nan,np.nan
    edges=np.linspace(lo,hi,rows+1); v=np.zeros(rows)
    for _,r in d.iterrows():
        a,b=float(r.low),float(r.high); vol=float(r.volume) if np.isfinite(r.volume) else 0
        if vol<=0:continue
        if b<=a:
            j=max(0,min(rows-1,np.searchsorted(edges,float(r.close),side='right')-1)); v[j]+=vol; continue
        ov=np.maximum(0,np.minimum(edges[1:],b)-np.maximum(edges[:-1],a)); s=ov.sum()
        if s:v+=vol*ov/s
    i=int(np.argmax(v)); target=v.sum()*va/100; l=r=i; cum=v[i]
    while cum<target and (l>0 or r<rows-1):
        lv=v[l-1] if l>0 else -1; rv=v[r+1] if r<rows-1 else -1
        if rv>=lv and r<rows-1:r+=1; cum+=v[r]
        elif l>0:l-=1; cum+=v[l]
        else:break
    centers=(edges[:-1]+edges[1:])/2
    return float(centers[i]),float(edges[r+1]),float(edges[l])

def levels(d,rows,va):
    x=d.copy(); x['s']=[sess(t) for t in x.index]; ss=sorted(x.s.unique()); out={}
    for i in range(1,len(ss)):
        p,c=ss[i-1],ss[i]; q=x[x.s==p]
        poc,vah,val=frvp(q,rows,va)
        if np.isfinite(poc):out[c]={'prev':p,'POC':poc,'VAH':vah,'VAL':val}
    return out

def backtest(d,lv,setup,start,risk,buf,maxbars):
    x=d.copy(); x['s']=[sess(t) for t in x.index]; x['n']=np.arange(len(x)); bal=float(start); peak=bal; dd=0; active=None; used=set(); trades=[]; eq=[]
    for s,g in x.groupby('s',sort=True):
        if s not in lv:continue
        L=lv[s]; poc,vah,val=L['POC'],L['VAH'],L['VAL']; active=None
        for pos in range(len(g)):
            ts=g.index[pos]; z=g.iloc[pos]; o,h,l,c=map(float,[z.open,z.high,z.low,z.close]); n=int(z.n)
            if active:
                side=active['side']; hs=l<=active['sl'] if side=='LONG' else h>=active['sl']; ht=h>=active['tp'] if side=='LONG' else l<=active['tp']; reason=None; ex=None
                if hs:reason,ex='SL',active['sl']
                elif ht:reason,ex='TP',active['tp']
                elif n-active['bar']>=maxbars:reason,ex='TIME',c
                if reason:
                    pnl=(ex-active['entry'])*active['qty'] if side=='LONG' else (active['entry']-ex)*active['qty']; r=pnl/active['rm']; bal+=pnl; peak=max(peak,bal); dd=max(dd,peak-bal)
                    trades.append({**active,'Exit Time':ts,'Exit':ex,'Result':'WIN' if pnl>0 else 'LOSS' if pnl<0 else 'FLAT','R':r,'PnL':pnl,'Balance':bal,'Exit Reason':reason}); active=None
            eq.append((ts,bal))
            if active or pos>=len(g)-1:continue
            prev=g.iloc[pos-1] if pos else None; cand=[]
            if setup in ('All','POC Bounce') and (s,'POC Bounce') not in used and l<=poc<=h:
                if c>poc and c>o and vah>c:c0=c; sl=l-buf; tp=vah; cand.append(('POC Bounce','LONG',c0,sl,tp,(s,'POC Bounce')) if c0>sl else None)
                elif c<poc and c<o and val<c:c0=c; sl=h+buf; tp=val; cand.append(('POC Bounce','SHORT',c0,sl,tp,(s,'POC Bounce')) if sl>c0 else None)
            if setup in ('All','POC Reversal') and prev is not None and (s,'POC Reversal') not in used:
                pc=float(prev.close)
                if pc<poc and c>poc and c>o:sl=l-buf;rm0=c-sl; cand.append(('POC Reversal','LONG',c,sl,c+rm0,(s,'POC Reversal')) if rm0>0 else None)
                elif pc>poc and c<poc and c<o:sl=h+buf;rm0=sl-c; cand.append(('POC Reversal','SHORT',c,sl,c-rm0,(s,'POC Reversal')) if rm0>0 else None)
            cand=[q for q in cand if q];
            if not cand:continue
            cand.sort(key=lambda q:0 if q[0]=='POC Reversal' else 1); setup0,side,en,sl,tp,key=cand[0]; rmoney=bal*risk/100; rp=abs(en-sl); qty=rmoney/rp if rp else 0
            if qty<=0:continue
            active={'Entry Time':ts,'Setup':setup0,'Side':side,'Entry':en,'SL':sl,'TP':tp,'qty':qty,'rm':rmoney,'bar':n,'POC':poc,'VAH':vah,'VAL':val,'Previous Session':str(L['prev'])}; used.add(key)
        if active:
            ts=g.index[-1]; ex=float(g.iloc[-1].close); side=active['Side']; pnl=(ex-active['Entry'])*active['qty'] if side=='LONG' else (active['Entry']-ex)*active['qty']; r=pnl/active['rm']; bal+=pnl; peak=max(peak,bal); dd=max(dd,peak-bal); trades.append({**active,'Exit Time':ts,'Exit':ex,'Result':'WIN' if pnl>0 else 'LOSS' if pnl<0 else 'FLAT','R':r,'PnL':pnl,'Balance':bal,'Exit Reason':'SESSION_CLOSE'}); eq.append((ts,bal))
    return pd.DataFrame(trades),pd.Series([v for _,v in eq],index=[t for t,_ in eq]),dd

st.title('FRVP POC • Delta V13 Backtester'); st.caption('15M analysis → 1M execution • POC Bounce + POC Reversal only')
with st.sidebar:
    try: ps=products(); syms=gold_symbols(ps)
    except Exception as e: ps=[]; syms=[]; st.error(f'Delta products error: {e}')
    symbol=st.selectbox('Delta gold product',syms or ['XAUUSD'])
    days=st.selectbox('History (days)',[3,5,7,10,14,20],index=2)
    rows=st.number_input('FRVP Row Size',20,200,60,5); va=st.number_input('Value Area %',50.,90.,70.,1.)
    setup=st.selectbox('Setup',['All','POC Bounce','POC Reversal']); buf=st.number_input('SL Buffer',0.,10.,.20,.05); maxbars=st.number_input('Max 1M bars',5,720,36,5)
    startbal=st.number_input('Starting Balance ($)',100.,1_000_000.,5000.,100.); risk=st.number_input('Risk per trade (%)',.1,5.,.5,.1)
    run=st.button('🚀 Run Delta Backtest',type='primary',use_container_width=True)
if run:
    now=datetime.now(timezone.utc); stime=now-timedelta(days=int(days))
    with st.spinner('Downloading Delta 15M + 1M candles...'):
        try:d15=candles(symbol,'15m',stime.timestamp(),now.timestamp()); d1=candles(symbol,'1m',stime.timestamp(),now.timestamp())
        except Exception as e:st.error(f'Delta data error: {e}');st.stop()
    if d15.empty or d1.empty:st.error('No historical data returned.');st.stop()
    lv=levels(d15,int(rows),float(va)); tr,eq,dd=backtest(d1,lv,setup,startbal,risk,buf,int(maxbars)); n=len(tr); wins=int((tr.Result=='WIN').sum()) if n else 0; gp=tr.loc[tr.PnL>0,'PnL'].sum() if n else 0; gl=-tr.loc[tr.PnL<0,'PnL'].sum() if n else 0; pf=gp/gl if gl else np.inf; netr=tr.R.sum() if n else 0; ending=float(tr.Balance.iloc[-1]) if n else startbal
    c=st.columns(7); c[0].metric('Trades',n); c[1].metric('Win Rate',f'{wins/n*100:.1f}%' if n else '0.0%'); c[2].metric('Net R',f'{netr:.2f}R'); c[3].metric('Avg R',f'{tr.R.mean():.2f}R' if n else '0.00R'); c[4].metric('Profit Factor','∞' if np.isinf(pf) else f'{pf:.2f}'); c[5].metric('Ending Balance',f'${ending:,.2f}'); c[6].metric('Max DD',f'${dd:,.2f}')
    st.info('No look-ahead: POC/VAH/VAL are calculated from the previous completed session using 15M candles, then frozen while the next session is stepped through 1M candles. Same-candle SL+TP = SL first.')
    if n:
        st.subheader('Performance by Setup'); s=tr.groupby('Setup').agg(Trades=('Setup','size'),Wins=('Result',lambda x:(x=='WIN').sum()),Losses=('Result',lambda x:(x=='LOSS').sum()),Net_R=('R','sum'),Avg_R=('R','mean'),PnL=('PnL','sum')).reset_index(); s['Win Rate %']=s.Wins/s.Trades*100; st.dataframe(s.round(2),use_container_width=True,hide_index=True)
        st.subheader('Equity Curve'); fig=go.Figure(go.Scatter(x=eq.index,y=eq.values,mode='lines',name='Balance')); fig.update_layout(height=340,xaxis_title='Time',yaxis_title='Balance ($)'); st.plotly_chart(fig,use_container_width=True)
        st.subheader('Trade Log'); disp=tr.drop(columns=['qty','rm','bar'],errors='ignore').copy(); st.dataframe(disp.round(3),use_container_width=True,hide_index=True); st.download_button('⬇️ Download CSV',tr.to_csv(index=False).encode(), 'frvp_delta_v13_trades.csv','text/csv',use_container_width=True)
    else:st.warning('No completed trades matched these settings.')
    st.subheader('Frozen Previous-Session FRVP'); rows2=[{'Execution Session':str(k),**{'Previous Session':str(v['prev']),'POC':v['POC'],'VAH':v['VAH'],'VAL':v['VAL']}} for k,v in lv.items()]; st.dataframe(pd.DataFrame(rows2).round(3),use_container_width=True,hide_index=True)
else:st.info('Select the Delta gold product and click Run Delta Backtest. Public historical candles require no API key.')
