'use strict';
(function(root){
function finiteArray(a,n,label){if(!Array.isArray(a)||a.length!==n||a.some(v=>typeof v!=='number'||!Number.isFinite(v)))throw Error(label+' must contain '+n+' finite numbers');}
function cholesky(c){const d=c.length,L=Array.from({length:d},()=>Array(d).fill(0));for(let i=0;i<d;i++){finiteArray(c[i],d,'covariance row');for(let j=0;j<=i;j++){if(Math.abs(c[i][j]-c[j][i])>1e-10)throw Error('covariance must be symmetric');let s=c[i][j];for(let h=0;h<j;h++)s-=L[i][h]*L[j][h];if(i===j){if(!(s>0))throw Error('covariance must be positive definite');L[i][j]=Math.sqrt(s);}else L[i][j]=s/L[j][j];}}return L;}
function infer(raw,model,mean,scale){
 const d=mean.length,k=model.weights.length;finiteArray(raw,d,'input');finiteArray(mean,d,'mean');finiteArray(scale,d,'scale');if(scale.some(v=>v<=0))throw Error('scale must be positive');finiteArray(model.weights,k,'weights');if(!k||model.weights.some(w=>w<=0)||Math.abs(model.weights.reduce((s,v)=>s+v,0)-1)>1e-10||model.means.length!==k||model.covariances.length!==k)throw Error('invalid model');
 const z=raw.map((v,i)=>(v-mean[i])/scale[i]);if(z.some(v=>!Number.isFinite(v)))throw Error('input exceeds numerical range');
 const logs=model.weights.map((w,j)=>{finiteArray(model.means[j],d,'component mean');if(model.covariances[j].length!==d)throw Error('covariance dimension differs');const L=cholesky(model.covariances[j]),a=Array(d).fill(0);for(let i=0;i<d;i++){let s=z[i]-model.means[j][i];for(let h=0;h<i;h++)s-=L[i][h]*a[h];a[i]=s/L[i][i];}return Math.log(w)-.5*(d*Math.log(2*Math.PI)+2*L.reduce((s,row,i)=>s+Math.log(row[i]),0)+a.reduce((s,v)=>s+v*v,0));});
 if(logs.some(v=>!Number.isFinite(v)))throw Error('input exceeds numerical range');const top=Math.max(...logs),a=logs.map(v=>Math.exp(v-top)),sum=a.reduce((s,v)=>s+v,0),logDensity=top+Math.log(sum),responsibilities=a.map(v=>v/sum);
 return {standardized:z,componentLogs:logs,logDensity,logDensityOriginal:logDensity-scale.reduce((s,v)=>s+Math.log(v),0),responsibilities,entropy:-responsibilities.reduce((s,v)=>s+(v>0?v*Math.log(v):0),0)};
}
const api={infer,cholesky};if(typeof module!=='undefined'&&module.exports)module.exports=api;root.MixtureInference=api;
})(globalThis);
