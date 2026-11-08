'use strict';
const test=require('node:test'),assert=require('node:assert/strict'),fs=require('node:fs'),path=require('node:path'),c=require('../docs/conditional.js');
require('../docs/data.js');const data=globalThis.MIXTURE_DATA,refs=JSON.parse(fs.readFileSync(path.join(__dirname,'../mixturelab/artifacts/conditional-reference.json'),'utf8'));
function close(a,b,atol=2e-9){if(Array.isArray(a)){assert.ok(Array.isArray(b));assert.equal(a.length,b.length);a.forEach((v,i)=>close(v,b[i],atol));}else if(typeof a==='object'&&a){assert.deepEqual(Object.keys(a).sort(),Object.keys(b).sort());for(const k of Object.keys(a))close(a[k],b[k],atol);}else if(typeof a==='number'){assert.ok(Number.isFinite(b));assert.ok(Math.abs(a-b)<=atol*(1+Math.abs(a)),`${a} != ${b}`);}else assert.deepEqual(a,b);}
test('all160 complete native conditional references, target log densities and controls',()=>{
 let n=0;for(const ref of refs.references){const source=data.models.find(m=>m.id===ref.id),r=ref.conditional,a=c.conditionRaw(source.model,r.observed,r.values,data.mean,data.scale);close(r,a);close(ref.log_density,c.logDensity(a,[ref.target])[0]);n++;}assert.equal(n,160);
 for(const v of Object.values(refs.controls)){const r=v.conditional||v.after;close(c.condition(v.model,r.observed,r.values),r);}
});
test('scalar Gaussian oracle, no mutation, zero dimension, invalid and underflow controls',()=>{
 const m={weights:[1],means:[[2,-1]],covariances:[[[4,1.5],[1.5,3]]]};const before=JSON.stringify(m);for(const x of [-5,0,2,9]){const r=c.condition(m,[0],[x]),mu=-1+1.5/4*(x-2),v=3-1.5**2/4;close(r.mean,[mu],1e-14);close(r.covariance,[[v]],1e-14);close(c.logDensity(r,[[0]])[0],-.5*(Math.log(2*Math.PI*v)+mu**2/v),1e-14);}assert.equal(JSON.stringify(m),before);
 for(const[obs,values]of[[[true],[0]],[[0,0],[0,0]],[[-1],[0]],[[2],[0]],[[0],[NaN]],[[0],[]],[[0],[1e308]],[[0],[true]]])assert.throws(()=>c.condition(m,obs,values));assert.equal(c.condition(m,[],[]).log_evidence,0);assert.deepEqual(c.logDensity(c.condition(m,[1,0],[0,0]),[[],[]]),[0,0]);
 const under={weights:[.5,.5],means:[[0,0],[100,100]],covariances:[[[1,0],[0,1]],[[1,0],[0,1]]]};const r=c.condition(under,[0],[0]);assert.equal(r.weights[1],0);close(c.logDensity(r,[[100]])[0],-5000-.5*Math.log(2*Math.PI)+Math.log(2),1e-12);
});
