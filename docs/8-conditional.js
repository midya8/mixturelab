'use strict';
(function(root){
const base=typeof module!=='undefined'&&module.exports?require('./inference.js'):root.MixtureInference;
const finite=(a,n,label)=>{if(!Array.isArray(a)||a.length!==n||a.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error(label+' must be a matching finite number array');};
function indices(obs,d){if(!Array.isArray(obs)||obs.some(i=>!Number.isInteger(i)||i<0||i>=d)||new Set(obs).size!==obs.length)throw Error('observed must be distinct integer coordinate indices');return Array.from({length:d},(_,i)=>i).filter(i=>!obs.includes(i));}
function solve(L,b){const d=b.length,z=Array(d).fill(0),out=Array(d).fill(0);for(let i=0;i<d;i++){let s=b[i];for(let j=0;j<i;j++)s-=L[i][j]*z[j];z[i]=s/L[i][i];}for(let i=d-1;i>=0;i--){let s=z[i];for(let j=i+1;j<d;j++)s-=L[j][i]*out[j];out[i]=s/L[i][i];}return out;}
function normal(x,mean,cov){const d=x.length;if(!d)return 0;const L=base.cholesky(cov),z=Array(d).fill(0);for(let i=0;i<d;i++){let s=x[i]-mean[i];for(let j=0;j<i;j++)s-=L[i][j]*z[j];z[i]=s/L[i][i];}const ll=-.5*(d*Math.log(2*Math.PI)+2*L.reduce((s,r,i)=>s+Math.log(r[i]),0)+z.reduce((s,v)=>s+v*v,0));if(!Number.isFinite(ll))throw Error('input exceeds finite numerical range');return ll;}
function lse(logs){const top=Math.max(...logs),out=top+Math.log(logs.reduce((s,v)=>s+Math.exp(v-top),0));if(!Number.isFinite(out))throw Error('density exceeds finite numerical range');return out;}
function condition(model,observed,values){
 if(!model||!Array.isArray(model.means)||!model.means.length||!Array.isArray(model.means[0]))throw Error('invalid model');const d=model.means[0].length,k=model.means.length;if(!d)throw Error('invalid dimension');const obs=observed?.slice(),free=indices(obs,d),n=free.length;finite(values,obs.length,'observed values');finite(model.weights,k,'weights');if(model.weights.some(w=>w<=0)||Math.abs(model.weights.reduce((s,w)=>s+w,0)-1)>1e-10||!Array.isArray(model.covariances)||model.covariances.length!==k)throw Error('invalid model');
 const means=[],covariances=[],logs=[];
 for(let j=0;j<k;j++){
  const mu=model.means[j],cov=model.covariances[j];finite(mu,d,'component mean');if(!Array.isArray(cov)||cov.length!==d)throw Error('invalid covariance');base.cholesky(cov);
  const coo=obs.map(i=>obs.map(h=>cov[i][h])),cmo=free.map(i=>obs.map(h=>cov[i][h])),cmm=free.map(i=>free.map(h=>cov[i][h])),L=base.cholesky(coo),delta=solve(L,values.map((v,i)=>v-mu[obs[i]])),gainT=cmo.map(row=>solve(L,row));
  const cm=free.map((i,a)=>mu[i]+cmo[a].reduce((s,v,b)=>s+v*delta[b],0)),cc=cmm.map((row,a)=>row.map((v,b)=>v-cmo[a].reduce((s,t,h)=>s+t*gainT[b][h],0)));
  for(let a=0;a<n;a++)for(let b=0;b<a;b++)cc[a][b]=cc[b][a]=.5*(cc[a][b]+cc[b][a]);if(n)base.cholesky(cc);
  means.push(cm);covariances.push(cc);logs.push(Math.log(model.weights[j])+normal(values,obs.map(i=>mu[i]),coo));
 }
 const normalizer=lse(logs),log_weights=logs.map(v=>v-normalizer),weights=log_weights.map(Math.exp),mean=free.map((_,a)=>means.reduce((s,row,j)=>s+weights[j]*row[a],0)),within_covariance=free.map((_,a)=>free.map((_,b)=>covariances.reduce((s,c,j)=>s+weights[j]*c[a][b],0))),between_covariance=free.map((_,a)=>free.map((_,b)=>means.reduce((s,row,j)=>s+weights[j]*(row[a]-mean[a])*(row[b]-mean[b]),0))),covariance=within_covariance.map((row,a)=>row.map((v,b)=>v+between_covariance[a][b]));
 const r={dimension:d,observed:obs,free,values:values.slice(),log_evidence:obs.length?normalizer:0,log_weights,weights,means,covariances,mean,within_covariance,between_covariance,covariance,coordinate_space:'standardized'};checkFinite(r);return r;
}
function checkFinite(r){for(const key of ['values','log_evidence','log_weights','weights','means','covariances','mean','within_covariance','between_covariance','covariance']){const a=Array.isArray(r[key])?r[key].flat(Infinity):[r[key]];if(a.some(v=>!Number.isFinite(v)))throw Error('conditional moments exceed finite numerical range');}}
function conditionRaw(model,obs,values,center,scale){const d=model.means[0].length;finite(center,d,'center');finite(scale,d,'scale');if(scale.some(v=>v<=0))throw Error('scale must be positive');indices(obs,d);finite(values,obs.length,'observed values');const r=condition(model,obs,values.map((v,i)=>(v-center[obs[i]])/scale[obs[i]])),free=r.free,s=free.map(i=>scale[i]);r.means=r.means.map(row=>row.map((v,a)=>v*s[a]+center[free[a]]));r.mean=r.mean.map((v,a)=>v*s[a]+center[free[a]]);for(const key of ['within_covariance','between_covariance','covariance'])r[key]=r[key].map((row,a)=>row.map((v,b)=>v*s[a]*s[b]));r.covariances=r.covariances.map(c=>c.map((row,a)=>row.map((v,b)=>v*s[a]*s[b])));r.values=values.slice();r.log_evidence-=obs.reduce((v,i)=>v+Math.log(scale[i]),0);r.coordinate_space='original';checkFinite(r);return r;}
function logDensity(r,samples){if(!Array.isArray(samples)||!samples.length)throw Error('targets must be nonempty');return samples.map(x=>{finite(x,r.free.length,'target');if(!r.free.length)return 0;return lse(r.log_weights.map((w,j)=>w+normal(x,r.means[j],r.covariances[j])));});}
const api={condition,conditionRaw,logDensity};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.MixtureConditional=api;
})(globalThis);
